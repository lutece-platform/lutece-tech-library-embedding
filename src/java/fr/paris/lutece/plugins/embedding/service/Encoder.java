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
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * Turns a text into a normalized vector with an ONNX model and its Hugging Face tokenizer.
 */
final class Encoder implements AutoCloseable
{
    private static final String INPUT_IDS = "input_ids";
    private static final String INPUT_MASK = "attention_mask";
    private static final String INPUT_TOKEN_TYPES = "token_type_ids";

    private final OrtEnvironment _env;
    private final OrtSession _session;
    private final HuggingFaceTokenizer _tokenizer;
    private final boolean _bMeanPooling;
    private final boolean _bTokenTypes;

    /**
     * Loads a model and its tokenizer.
     *
     * @param model
     *            the ONNX model file
     * @param tokenizer
     *            the tokenizer.json file
     * @param descriptor
     *            the description of the model
     * @param nThreads
     *            the number of threads of one encoding
     * @throws OrtException
     *             if the model cannot be loaded
     * @throws IOException
     *             if the tokenizer cannot be loaded
     */
    Encoder( Path model, Path tokenizer, ModelDescriptor descriptor, int nThreads ) throws OrtException, IOException
    {
        _env = OrtEnvironment.getEnvironment( );
        _tokenizer = HuggingFaceTokenizer.builder( ).optTokenizerPath( tokenizer ).optMaxLength( descriptor.getMaxTokens( ) ).optTruncation( true )
                .optPadding( false ).optAddSpecialTokens( true ).build( );
        try ( OrtSession.SessionOptions options = new OrtSession.SessionOptions( ) )
        {
            options.setIntraOpNumThreads( nThreads );
            options.setInterOpNumThreads( 1 );
            _session = _env.createSession( model.toString( ), options );
        }
        catch( OrtException | RuntimeException e )
        {
            _tokenizer.close( );
            throw e;
        }
        _bMeanPooling = ModelDescriptor.POOLING_MEAN.equals( descriptor.getPooling( ) );
        _bTokenTypes = _session.getInputNames( ).contains( INPUT_TOKEN_TYPES );
    }

    /**
     * Returns the normalized vector of a text.
     *
     * @param strText
     *            the text, prefix included
     * @return the vector
     * @throws OrtException
     *             if the model fails
     */
    float [ ] encode( String strText ) throws OrtException
    {
        Encoding encoding = _tokenizer.encode( strText );
        long [ ] ids = encoding.getIds( );
        long [ ] mask = encoding.getAttentionMask( );
        Map<String, OnnxTensor> inputs = new HashMap<>( );
        try
        {
            inputs.put( INPUT_IDS, OnnxTensor.createTensor( _env, new long [ ] [ ] {
                    ids
            } ) );
            inputs.put( INPUT_MASK, OnnxTensor.createTensor( _env, new long [ ] [ ] {
                    mask
            } ) );
            if ( _bTokenTypes )
            {
                inputs.put( INPUT_TOKEN_TYPES, OnnxTensor.createTensor( _env, new long [ 1 ] [ ids.length] ) );
            }
            try ( OrtSession.Result result = _session.run( inputs ) )
            {
                float [ ] [ ] tokens = ( (float [ ] [ ] [ ]) result.get( 0 ).getValue( ) ) [0];
                return normalize( _bMeanPooling ? mean( tokens, mask ) : tokens [0].clone( ) );
            }
        }
        finally
        {
            inputs.values( ).forEach( OnnxTensor::close );
        }
    }

    /**
     * Returns the mean of the token vectors the mask keeps.
     *
     * @param tokens
     *            the token vectors
     * @param mask
     *            the attention mask
     * @return the mean vector
     */
    private static float [ ] mean( float [ ] [ ] tokens, long [ ] mask )
    {
        float [ ] vector = new float [ tokens [0].length];
        int nKept = 0;
        for ( int t = 0; t < tokens.length; t++ )
        {
            if ( mask [t] != 0 )
            {
                nKept++;
                for ( int i = 0; i < vector.length; i++ )
                {
                    vector [i] += tokens [t] [i];
                }
            }
        }
        for ( int i = 0; i < vector.length; i++ )
        {
            vector [i] /= Math.max( 1, nKept );
        }
        return vector;
    }

    /**
     * Scales a vector to a norm of 1.
     *
     * @param vector
     *            the vector, changed in place
     * @return the vector
     */
    private static float [ ] normalize( float [ ] vector )
    {
        double dNorm = 0;
        for ( float f : vector )
        {
            dNorm += f * f;
        }
        double dScale = dNorm == 0 ? 0 : 1 / Math.sqrt( dNorm );
        for ( int i = 0; i < vector.length; i++ )
        {
            vector [i] = (float) ( vector [i] * dScale );
        }
        return vector;
    }

    /**
     * Releases the model and the tokenizer.
     *
     * @throws OrtException
     *             if the model cannot be released
     */
    @Override
    public void close( ) throws OrtException
    {
        try
        {
            _tokenizer.close( );
        }
        finally
        {
            _session.close( );
        }
    }
}
