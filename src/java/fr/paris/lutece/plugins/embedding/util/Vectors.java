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
package fr.paris.lutece.plugins.embedding.util;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * Similarity and storage of embedding vectors.
 */
public final class Vectors
{
    private static final int FLOAT_BYTES = Float.BYTES;

    /**
     * Private constructor.
     */
    private Vectors( )
    {
    }

    /**
     * Returns the cosine similarity of two vectors of the same dimension.
     *
     * @param a
     *            a vector
     * @param b
     *            another vector
     * @return the similarity, between -1 and 1, 0 when a vector is zero
     */
    public static double similarity( float [ ] a, float [ ] b )
    {
        if ( a.length != b.length )
        {
            throw new IllegalArgumentException( "Vectors of dimensions " + a.length + " and " + b.length );
        }
        double dDot = 0;
        double dNormA = 0;
        double dNormB = 0;
        for ( int i = 0; i < a.length; i++ )
        {
            dDot += a [i] * b [i];
            dNormA += a [i] * a [i];
            dNormB += b [i] * b [i];
        }
        return ( dNormA == 0 || dNormB == 0 ) ? 0 : dDot / Math.sqrt( dNormA * dNormB );
    }

    /**
     * Returns the candidates most similar to a query, most similar first.
     *
     * @param <K>
     *            the type of the candidate keys
     * @param query
     *            the query vector
     * @param candidates
     *            the candidate vectors by key
     * @param nCount
     *            the maximum number of neighbours
     * @param dMinScore
     *            the minimum similarity of a neighbour
     * @return the neighbours
     */
    public static <K> List<Neighbour<K>> nearest( float [ ] query, Map<K, float [ ]> candidates, int nCount, double dMinScore )
    {
        List<Neighbour<K>> listNeighbours = new ArrayList<>( );
        for ( Map.Entry<K, float [ ]> candidate : candidates.entrySet( ) )
        {
            double dScore = similarity( query, candidate.getValue( ) );
            if ( dScore >= dMinScore )
            {
                listNeighbours.add( new Neighbour<>( candidate.getKey( ), dScore ) );
            }
        }
        listNeighbours.sort( Comparator.comparingDouble( Neighbour<K>::getScore ).reversed( ) );
        return listNeighbours.size( ) > nCount ? new ArrayList<>( listNeighbours.subList( 0, nCount ) ) : listNeighbours;
    }

    /**
     * Encodes a vector as bytes, four little-endian bytes per dimension.
     *
     * @param vector
     *            the vector
     * @return the bytes
     */
    public static byte [ ] toBytes( float [ ] vector )
    {
        ByteBuffer buffer = ByteBuffer.allocate( vector.length * FLOAT_BYTES ).order( ByteOrder.LITTLE_ENDIAN );
        buffer.asFloatBuffer( ).put( vector );
        return buffer.array( );
    }

    /**
     * Decodes a vector encoded by {@link #toBytes(float[])}.
     *
     * @param bytes
     *            the bytes
     * @return the vector
     */
    public static float [ ] fromBytes( byte [ ] bytes )
    {
        if ( bytes.length % FLOAT_BYTES != 0 )
        {
            throw new IllegalArgumentException( "Not a vector: " + bytes.length + " bytes" );
        }
        float [ ] vector = new float [ bytes.length / FLOAT_BYTES];
        ByteBuffer.wrap( bytes ).order( ByteOrder.LITTLE_ENDIAN ).asFloatBuffer( ).get( vector );
        return vector;
    }
}
