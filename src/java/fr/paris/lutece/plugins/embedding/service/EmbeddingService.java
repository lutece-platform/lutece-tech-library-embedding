/*
 * Copyright (c) 2002-2026, City of Paris
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice
 *     and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice
 *     and the following disclaimer in the documentation and/or other materials
 *     provided with the distribution.
 *
 *  3. Neither the name of 'Mairie de Paris' nor 'Lutece' nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * License 1.0
 */
package fr.paris.lutece.plugins.embedding.service;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import ai.onnxruntime.OrtException;
import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

/**
 * Turns texts into vectors with an embedding model run in the JVM. The model is loaded at the first call; when it
 * cannot be loaded, the service is not available and the callers fall back on another search.
 */
@ApplicationScoped
public class EmbeddingService
{
    private static final String RESOURCE_BASE = "fr/paris/lutece/plugins/embedding/models/";
    private static final String FILE_MODEL = "model.onnx";
    private static final String FILE_TOKENIZER = "tokenizer.json";
    private static final String TEMP_PREFIX = "lutece-embedding-";
    private static final String CPU_INFO = "/proc/cpuinfo";
    private static final String CGROUP_V2_LIMIT = "/sys/fs/cgroup/memory.max";
    private static final String CGROUP_V1_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes";
    private static final String UNKNOWN = "unknown";
    private static final long MEBIBYTE = 1024L * 1024L;
    private static final String PROBE_TEXT = "embedding probe";
    private static final Logger LOGGER = LogManager.getLogger( "lutece.embedding" );

    private String _strModelName;
    private Optional<String> _strModelPath;
    private int _nThreads;
    private volatile boolean _bLoaded;
    private Encoder _encoder;
    private ModelDescriptor _descriptor;

    /**
     * Constructor for the CDI proxy.
     */
    EmbeddingService( )
    {
    }

    /**
     * Creates the service.
     *
     * @param strModelName
     *            the name of the model on the classpath, as packaged by a library-embedding model artifact
     * @param strModelPath
     *            a directory holding model.onnx, tokenizer.json and model.properties, used instead of the classpath
     * @param nThreads
     *            the number of threads of one encoding
     */
    @Inject
    public EmbeddingService( @ConfigProperty( name = "embedding.model.name", defaultValue = "granite-r2" ) String strModelName,
            @ConfigProperty( name = "embedding.model.path" ) Optional<String> strModelPath,
            @ConfigProperty( name = "embedding.threads", defaultValue = "2" ) int nThreads )
    {
        _strModelName = strModelName;
        _strModelPath = strModelPath;
        _nThreads = nThreads;
    }

    /**
     * Tells whether a model is loaded, loading it at the first call.
     *
     * @return true when texts can be turned into vectors
     */
    public boolean isAvailable( )
    {
        return getEncoder( ) != null;
    }

    /**
     * Returns the identifier of the loaded model, to store with each vector.
     *
     * @return the identifier
     */
    public String getModelId( )
    {
        requireEncoder( );
        return _descriptor.getId( );
    }

    /**
     * Returns the dimension of the vectors of the loaded model.
     *
     * @return the dimension
     */
    public int getDimensions( )
    {
        requireEncoder( );
        return _descriptor.getDimensions( );
    }

    /**
     * Returns the vector of a query, such as a title being typed.
     *
     * @param strText
     *            the text
     * @return the normalized vector
     */
    public float [ ] embedQuery( String strText )
    {
        return encode( requireEncoder( ), _descriptor.getQueryPrefix( ), strText );
    }

    /**
     * Returns the vector of a stored document, such as a submitted idea.
     *
     * @param strText
     *            the text
     * @return the normalized vector
     */
    public float [ ] embedDocument( String strText )
    {
        return encode( requireEncoder( ), _descriptor.getDocumentPrefix( ), strText );
    }

    /**
     * Returns the vectors of several stored documents, in the same order.
     *
     * @param listTexts
     *            the texts
     * @return the normalized vectors
     */
    public List<float [ ]> embedDocuments( List<String> listTexts )
    {
        Encoder encoder = requireEncoder( );
        List<float [ ]> listVectors = new ArrayList<>( listTexts.size( ) );
        for ( String strText : listTexts )
        {
            listVectors.add( encode( encoder, _descriptor.getDocumentPrefix( ), strText ) );
        }
        return listVectors;
    }

    /**
     * Releases the model.
     */
    @PreDestroy
    public synchronized void close( )
    {
        if ( _encoder != null )
        {
            try
            {
                _encoder.close( );
            }
            catch( OrtException e )
            {
                LOGGER.error( "embedding- error releasing the model", e );
            }
            _encoder = null;
        }
    }

    /**
     * Encodes a prefixed text.
     *
     * @param encoder
     *            the encoder
     * @param strPrefix
     *            the prefix of the model
     * @param strText
     *            the text
     * @return the normalized vector
     */
    private static float [ ] encode( Encoder encoder, String strPrefix, String strText )
    {
        if ( strText == null || strText.isBlank( ) )
        {
            throw new IllegalArgumentException( "No text to embed" );
        }
        try
        {
            return encoder.encode( strPrefix + strText );
        }
        catch( OrtException e )
        {
            throw new IllegalStateException( "Embedding failed", e );
        }
    }

    /**
     * Returns the encoder, or fails when no model could be loaded.
     *
     * @return the encoder
     */
    private Encoder requireEncoder( )
    {
        Encoder encoder = getEncoder( );
        if ( encoder == null )
        {
            throw new IllegalStateException( "No embedding model is available" );
        }
        return encoder;
    }

    /**
     * Returns the encoder, loading the model once.
     *
     * @return the encoder, null when the model cannot be loaded
     */
    private Encoder getEncoder( )
    {
        if ( !_bLoaded )
        {
            synchronized( this )
            {
                if ( !_bLoaded )
                {
                    _encoder = load( );
                    _bLoaded = true;
                }
            }
        }
        return _encoder;
    }

    /**
     * Loads the model from its directory or from the classpath, and logs what the environment offers.
     *
     * @return the encoder, null when the model cannot be loaded
     */
    private Encoder load( )
    {
        long lStart = System.currentTimeMillis( );
        String strSource = _strModelPath.filter( s -> !s.isBlank( ) ).map( s -> "directory " + s ).orElse( "classpath " + RESOURCE_BASE + _strModelName );
        try
        {
            Encoder encoder = _strModelPath.filter( s -> !s.isBlank( ) ).isPresent( ) ? loadFromDirectory( Path.of( _strModelPath.get( ) ) )
                    : loadFromClasspath( );
            long lLoaded = System.currentTimeMillis( );
            encoder.encode( _descriptor.getDocumentPrefix( ) + PROBE_TEXT );
            LOGGER.info( "embedding- model {} loaded from {} in {} ms: {} dimensions, {} threads | first encoding {} ms | {}", _descriptor.getId( ),
                    strSource, lLoaded - lStart, _descriptor.getDimensions( ), _nThreads, System.currentTimeMillis( ) - lLoaded, describeEnvironment( ) );
            return encoder;
        }
        catch( IOException | OrtException | RuntimeException | LinkageError e )
        {
            LOGGER.error( "embedding- no model loaded from {} | {}", strSource, describeEnvironment( ), e );
            return null;
        }
    }

    /**
     * Loads a model from a directory.
     *
     * @param directory
     *            the directory holding model.onnx, tokenizer.json and model.properties
     * @return the encoder
     * @throws IOException
     *             if a file cannot be read
     * @throws OrtException
     *             if the model cannot be loaded
     */
    private Encoder loadFromDirectory( Path directory ) throws IOException, OrtException
    {
        try ( InputStream input = Files.newInputStream( directory.resolve( ModelDescriptor.FILE_NAME ) ) )
        {
            _descriptor = new ModelDescriptor( input );
        }
        return new Encoder( directory.resolve( FILE_MODEL ), directory.resolve( FILE_TOKENIZER ), _descriptor, _nThreads );
    }

    /**
     * Loads a model packaged on the classpath, copied to a temporary directory removed once loaded.
     *
     * @return the encoder
     * @throws IOException
     *             if a resource is missing or cannot be copied
     * @throws OrtException
     *             if the model cannot be loaded
     */
    private Encoder loadFromClasspath( ) throws IOException, OrtException
    {
        String strBase = RESOURCE_BASE + _strModelName + "/";
        Path directory = Files.createTempDirectory( TEMP_PREFIX );
        try
        {
            for ( String strFile : new String [ ] {
                    ModelDescriptor.FILE_NAME, FILE_MODEL, FILE_TOKENIZER
            } )
            {
                try ( InputStream input = EmbeddingService.class.getClassLoader( ).getResourceAsStream( strBase + strFile ) )
                {
                    if ( input == null )
                    {
                        throw new IOException( "No resource " + strBase + strFile + " on the classpath" );
                    }
                    Files.copy( input, directory.resolve( strFile ), StandardCopyOption.REPLACE_EXISTING );
                }
            }
            return loadFromDirectory( directory );
        }
        finally
        {
            for ( String strFile : new String [ ] {
                    ModelDescriptor.FILE_NAME, FILE_MODEL, FILE_TOKENIZER
            } )
            {
                Files.deleteIfExists( directory.resolve( strFile ) );
            }
            Files.deleteIfExists( directory );
        }
    }

    /**
     * Describes what the environment offers to the model: the AVX2 instructions the quantized models need, the memory
     * limit of the container, the heap and the temporary directory where the native libraries are extracted.
     *
     * @return the description
     */
    static String describeEnvironment( )
    {
        return "cpu avx2 " + readCpuAvx2( ) + ", container memory limit " + readMemoryLimit( ) + ", max heap "
                + Runtime.getRuntime( ).maxMemory( ) / MEBIBYTE + " MiB, tmp " + System.getProperty( "java.io.tmpdir" );
    }

    /**
     * Tells whether the CPU offers AVX2.
     *
     * @return yes, no or unknown
     */
    private static String readCpuAvx2( )
    {
        try
        {
            return Files.readString( Path.of( CPU_INFO ) ).toLowerCase( Locale.ROOT ).contains( " avx2" ) ? "yes" : "no";
        }
        catch( IOException | RuntimeException e )
        {
            return UNKNOWN;
        }
    }

    /**
     * Returns the memory limit of the container.
     *
     * @return the limit in MiB, none, or unknown
     */
    private static String readMemoryLimit( )
    {
        for ( String strFile : new String [ ] {
                CGROUP_V2_LIMIT, CGROUP_V1_LIMIT
        } )
        {
            try
            {
                String strValue = Files.readString( Path.of( strFile ) ).trim( );
                return "max".equals( strValue ) ? "none" : Long.parseLong( strValue ) / MEBIBYTE + " MiB";
            }
            catch( IOException | RuntimeException e )
            {
                LOGGER.debug( "embedding- no memory limit in {}", strFile );
            }
        }
        return UNKNOWN;
    }
}
