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
import java.util.Properties;

/**
 * Description of a model, read from its model.properties file.
 */
public final class ModelDescriptor
{
    public static final String FILE_NAME = "model.properties";
    public static final String POOLING_CLS = "CLS";
    public static final String POOLING_MEAN = "MEAN";

    private static final String KEY_ID = "id";
    private static final String KEY_DIMENSIONS = "dimensions";
    private static final String KEY_POOLING = "pooling";
    private static final String KEY_QUERY_PREFIX = "queryPrefix";
    private static final String KEY_DOCUMENT_PREFIX = "documentPrefix";
    private static final String KEY_MAX_TOKENS = "maxTokens";
    private static final String DEFAULT_MAX_TOKENS = "512";

    private final String _strId;
    private final int _nDimensions;
    private final String _strPooling;
    private final String _strQueryPrefix;
    private final String _strDocumentPrefix;
    private final int _nMaxTokens;

    /**
     * Reads a descriptor.
     *
     * @param input
     *            the content of a model.properties file
     * @throws IOException
     *             if the content cannot be read or is incomplete
     */
    public ModelDescriptor( InputStream input ) throws IOException
    {
        Properties properties = new Properties( );
        properties.load( input );
        _strId = required( properties, KEY_ID );
        _nDimensions = Integer.parseInt( required( properties, KEY_DIMENSIONS ) );
        _strPooling = required( properties, KEY_POOLING ).toUpperCase( );
        if ( !POOLING_CLS.equals( _strPooling ) && !POOLING_MEAN.equals( _strPooling ) )
        {
            throw new IOException( "Unknown pooling " + _strPooling );
        }
        _strQueryPrefix = properties.getProperty( KEY_QUERY_PREFIX, "" );
        _strDocumentPrefix = properties.getProperty( KEY_DOCUMENT_PREFIX, "" );
        _nMaxTokens = Integer.parseInt( properties.getProperty( KEY_MAX_TOKENS, DEFAULT_MAX_TOKENS ).trim( ) );
    }

    /**
     * Returns a required property.
     *
     * @param properties
     *            the properties
     * @param strKey
     *            the key
     * @return the trimmed value
     * @throws IOException
     *             if the value is missing
     */
    private static String required( Properties properties, String strKey ) throws IOException
    {
        String strValue = properties.getProperty( strKey );
        if ( strValue == null || strValue.isBlank( ) )
        {
            throw new IOException( "Missing " + strKey + " in " + FILE_NAME );
        }
        return strValue.trim( );
    }

    /**
     * Returns the model identifier, stored with each vector.
     *
     * @return the identifier
     */
    public String getId( )
    {
        return _strId;
    }

    /**
     * Returns the dimension of the vectors.
     *
     * @return the dimension
     */
    public int getDimensions( )
    {
        return _nDimensions;
    }

    /**
     * Returns the pooling of the token vectors, CLS or MEAN.
     *
     * @return the pooling
     */
    public String getPooling( )
    {
        return _strPooling;
    }

    /**
     * Returns the prefix of a query text.
     *
     * @return the prefix, empty when none
     */
    public String getQueryPrefix( )
    {
        return _strQueryPrefix;
    }

    /**
     * Returns the prefix of a document text.
     *
     * @return the prefix, empty when none
     */
    public String getDocumentPrefix( )
    {
        return _strDocumentPrefix;
    }

    /**
     * Returns the number of tokens beyond which a text is cut.
     *
     * @return the maximum number of tokens
     */
    public int getMaxTokens( )
    {
        return _nMaxTokens;
    }
}
