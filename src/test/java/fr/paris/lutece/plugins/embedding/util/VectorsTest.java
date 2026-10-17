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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

/**
 * Tests the vector utilities.
 */
public class VectorsTest
{
    /**
     * Checks the cosine similarity of identical, opposite, orthogonal and zero vectors.
     */
    @Test
    public void testSimilarity( )
    {
        assertEquals( 1, Vectors.similarity( new float [ ] { 1, 2, 3 }, new float [ ] { 2, 4, 6 } ), 1e-6 );
        assertEquals( -1, Vectors.similarity( new float [ ] { 1, 0 }, new float [ ] { -1, 0 } ), 1e-6 );
        assertEquals( 0, Vectors.similarity( new float [ ] { 1, 0 }, new float [ ] { 0, 1 } ), 1e-6 );
        assertEquals( 0, Vectors.similarity( new float [ ] { 0, 0 }, new float [ ] { 0, 1 } ), 1e-6 );
        assertThrows( IllegalArgumentException.class, ( ) -> Vectors.similarity( new float [ ] { 1 }, new float [ ] { 1, 2 } ) );
    }

    /**
     * Checks that a vector survives its encoding as bytes.
     */
    @Test
    public void testBytesRoundTrip( )
    {
        float [ ] vector = { 0.25f, -1.5f, 3.75e-7f, Float.MIN_VALUE };
        byte [ ] bytes = Vectors.toBytes( vector );
        assertEquals( 16, bytes.length );
        assertArrayEquals( vector, Vectors.fromBytes( bytes ) );
        assertThrows( IllegalArgumentException.class, ( ) -> Vectors.fromBytes( new byte [ 5 ] ) );
    }

    /**
     * Checks that a vector is stored as little-endian float32, the layout of the vectors already stored.
     */
    @Test
    public void testBytesAreLittleEndian( )
    {
        assertArrayEquals( new byte [ ] { 0, 0, (byte) 0x80, 0x3F, 0, 0, 0, (byte) 0xC0 }, Vectors.toBytes( new float [ ] { 1, -2 } ) );
    }

    /**
     * Checks that the nearest neighbours come most similar first, above the minimum score, within the count.
     */
    @Test
    public void testNearest( )
    {
        Map<String, float [ ]> candidates = new LinkedHashMap<>( );
        candidates.put( "far", new float [ ] { 0, 1 } );
        candidates.put( "near", new float [ ] { 1, 0.1f } );
        candidates.put( "nearest", new float [ ] { 1, 0 } );
        candidates.put( "opposite", new float [ ] { -1, 0 } );

        List<Neighbour<String>> listNeighbours = Vectors.nearest( new float [ ] { 1, 0 }, candidates, 2, 0.5 );

        assertEquals( 2, listNeighbours.size( ) );
        assertEquals( "nearest", listNeighbours.get( 0 ).getKey( ) );
        assertEquals( "near", listNeighbours.get( 1 ).getKey( ) );
        assertEquals( 1, Vectors.nearest( new float [ ] { 1, 0 }, candidates, 5, 0.999 ).size( ) );
    }
}
