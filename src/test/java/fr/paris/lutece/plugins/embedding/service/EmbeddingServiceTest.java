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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvFileSource;

import fr.paris.lutece.plugins.embedding.util.Vectors;

/**
 * Tests the embedding service on the Granite R2 model of library-embedding-granite-r2.
 */
public class EmbeddingServiceTest
{
    private static final String [ ] TITLES = {
            "Avoir Claude Cowork pour consolider les tableaux des arrondissements",
            "Utiliser Copilot dans Word et Excel, pas seulement dans le navigateur",
            "Disposer d’un outil d’IA validé par la Ville, pour arrêter les comptes personnels",
            "Suivre une formation courte sur ce qu’on a le droit de faire avec l’IA", "Savoir quelles données on peut confier à une IA",
            "Équiper les salles de réunion d’un micro pour la transcription", "Remplacer les portables trop lents pour les outils d’IA",
            "Dicter un signalement depuis le terrain, sur le téléphone", "Partager nos consignes d’IA entre collègues du même service",
            "Interroger l’intranet et les notes de service en langage courant", "Obtenir des réponses sourcées pour les questions juridiques",
            "Préparer le procès-verbal du conseil d’arrondissement depuis l’enregistrement",
            "Trier la boîte mail partagée et proposer un brouillon de réponse",
            "Réécrire les courriers aux usagers en Facile à lire et à comprendre", "Sous-titrer automatiquement les vidéos publiées par la Ville",
            "Comparer deux versions d’un règlement et lister ce qui change", "Remplir la fiche d’intervention à partir d’une note vocale",
            "Regrouper les réponses libres des enquêtes de satisfaction", "Préparer un premier jet de délibération à partir des pièces du dossier",
            "Aider les agents d’accueil à répondre dans la langue des usagers"
    };

    private static final String [ ] REFORMULATIONS = {
            "Synthétiser automatiquement les fichiers Excel envoyés par les mairies d’arrondissement",
            "Avoir l’assistant Microsoft intégré à la suite Office", "Une IA officielle et sécurisée pour ne plus utiliser ChatGPT perso",
            "Former les agents aux règles d’usage de l’intelligence artificielle", "Quelles informations ai-je le droit de mettre dans ChatGPT ?",
            "Transcrire automatiquement les réunions grâce à un microphone", "Des ordinateurs plus puissants pour faire tourner l’IA",
            "Signaler un problème de voirie à la voix avec son smartphone", "Mutualiser les prompts efficaces au sein de l’équipe",
            "Un chatbot pour chercher dans les documents internes", "Une IA juridique qui cite ses sources",
            "Rédiger le compte rendu de séance à partir de l’audio", "Classer les mails de la boîte générique et suggérer des réponses",
            "Simplifier le langage des lettres envoyées aux habitants", "Générer les sous-titres des vidéos",
            "Voir les différences entre l’ancien et le nouveau texte d’un arrêté", "Compléter le rapport d’intervention en parlant au téléphone",
            "Analyser les commentaires des questionnaires usagers", "Rédiger un projet de délibération avec l’IA",
            "Traduire en direct pour les usagers étrangers au guichet"
    };

    private static EmbeddingService _service;

    /**
     * Loads the model once for the tests.
     */
    @BeforeAll
    public static void setUp( )
    {
        _service = new EmbeddingService( "granite-r2", Optional.empty( ), 2 );
    }

    /**
     * Releases the model.
     */
    @AfterAll
    public static void tearDown( )
    {
        _service.close( );
    }

    /**
     * Checks that the packaged model loads and describes itself.
     */
    @Test
    public void testModelLoads( )
    {
        assertTrue( _service.isAvailable( ) );
        assertEquals( 384, _service.getDimensions( ) );
        assertTrue( _service.getModelId( ).startsWith( "ibm-granite/granite-embedding-97m-multilingual-r2@" ) );
    }

    /**
     * Checks that a vector is normalized, of the model dimension, and the same at each call.
     */
    @Test
    public void testVectorIsStable( )
    {
        float [ ] vector = _service.embedDocument( TITLES [0] );
        assertEquals( 384, vector.length );
        assertEquals( 1, Vectors.similarity( vector, vector ), 1e-5 );
        double dNorm = 0;
        for ( float f : vector )
        {
            dNorm += f * f;
        }
        assertEquals( 1, Math.sqrt( dNorm ), 1e-4 );
        assertArrayEquals( vector, _service.embedDocument( TITLES [0] ) );
    }

    /**
     * Checks that the vectors match those sentence-transformers computes from the same model file with the pooling
     * the model declares, CLS; a mean pooling scores at most 0.86 against them.
     *
     * @param strText
     *            the text
     * @param strVector
     *            the reference vector, values separated by spaces
     */
    @ParameterizedTest
    @CsvFileSource( resources = "granite-r2-reference.txt", delimiter = '\t', maxCharsPerColumn = 8192 )
    public void testVectorsMatchTheReference( String strText, String strVector )
    {
        String [ ] values = strVector.split( " " );
        float [ ] reference = new float [ values.length];
        for ( int i = 0; i < values.length; i++ )
        {
            reference [i] = Float.parseFloat( values [i] );
        }
        assertEquals( 1, Vectors.similarity( reference, _service.embedDocument( strText ) ), 1e-3 );
    }

    /**
     * Checks that a blank text is refused.
     */
    @Test
    public void testBlankTextIsRefused( )
    {
        assertThrows( IllegalArgumentException.class, ( ) -> _service.embedQuery( "  " ) );
        assertThrows( IllegalArgumentException.class, ( ) -> _service.embedDocument( null ) );
    }

    /**
     * Checks that a missing model leaves the service unavailable instead of failing.
     */
    @Test
    public void testMissingModelIsUnavailable( )
    {
        EmbeddingService service = new EmbeddingService( "no-such-model", Optional.empty( ), 1 );
        assertFalse( service.isAvailable( ) );
        assertThrows( IllegalStateException.class, ( ) -> service.embedQuery( "texte" ) );
    }

    /**
     * Checks that concurrent calls give the same vectors as single calls.
     *
     * @throws Exception
     *             if a call fails
     */
    @Test
    public void testConcurrentCalls( ) throws Exception
    {
        List<float [ ]> listExpected = _service.embedDocuments( List.of( TITLES ) );
        ExecutorService executor = Executors.newFixedThreadPool( 4 );
        try
        {
            List<Future<List<float [ ]>>> listFutures = new ArrayList<>( );
            for ( int n = 0; n < 4; n++ )
            {
                listFutures.add( executor.submit( ( ) -> _service.embedDocuments( List.of( TITLES ) ) ) );
            }
            for ( Future<List<float [ ]>> future : listFutures )
            {
                List<float [ ]> listVectors = future.get( );
                for ( int i = 0; i < TITLES.length; i++ )
                {
                    assertArrayEquals( listExpected.get( i ), listVectors.get( i ) );
                }
            }
        }
        finally
        {
            executor.shutdown( );
        }
    }

    /**
     * Checks that a reformulated title finds the title it reformulates among the ideas of the reference set.
     */
    @Test
    public void testReformulationsFindTheirTitle( )
    {
        List<float [ ]> listTitles = _service.embedDocuments( List.of( TITLES ) );
        int nFirst = 0;
        int nTopThree = 0;
        for ( int q = 0; q < REFORMULATIONS.length; q++ )
        {
            float [ ] query = _service.embedQuery( REFORMULATIONS [q] );
            double dTarget = Vectors.similarity( query, listTitles.get( q ) );
            int nRank = 1;
            for ( int i = 0; i < listTitles.size( ); i++ )
            {
                if ( i != q && Vectors.similarity( query, listTitles.get( i ) ) > dTarget )
                {
                    nRank++;
                }
            }
            nFirst += nRank == 1 ? 1 : 0;
            nTopThree += nRank <= 3 ? 1 : 0;
        }
        System.out.println( "embedding- reference set: first " + nFirst + "/20, top three " + nTopThree + "/20" );
        assertTrue( nFirst >= 12, "first: " + nFirst + "/20" );
        assertTrue( nTopThree >= 15, "top three: " + nTopThree + "/20" );
    }
}
