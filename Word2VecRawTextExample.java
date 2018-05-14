package org.deeplearning4j.examples.nlp.word2vec;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import zemberek.morphology.ambiguity.Z3MarkovModelDisambiguator;
import zemberek.morphology.analysis.SentenceAnalysis;
import zemberek.morphology.analysis.WordAnalysis;
import zemberek.morphology.analysis.tr.TurkishMorphology;
import zemberek.morphology.analysis.tr.TurkishSentenceAnalyzer;

/**
 * Created by agibsonccc on 10/9/14.
 *
 * Neural net that processes text into wordvectors. See below url for an
 * in-depth explanation. https://deeplearning4j.org/word2vec.html
 * 
 * Edited by Umut Cetin for Turkish Morphological Disambiguation
 */
public class Word2VecRawTextExample {

	private static ArrayList<String> test_data;

	TurkishSentenceAnalyzer sentenceAnalyzer;

	private static Logger log = LoggerFactory.getLogger(Word2VecRawTextExample.class);

	// taken from zemberek examples : disambiguate sentences
	void analyzeAndDisambiguate(String sentence) throws IOException {

		SentenceAnalysis result = sentenceAnalyzer.analyze(sentence);

		// System.out.println("Before disambiguation.");
		// writeParseResult(result);

		System.out.println("\nAfter disambiguation.");
		sentenceAnalyzer.disambiguate(result);
		writeParseResult(result);

	}

	private void writeParseResult(SentenceAnalysis sentenceAnalysis) {
		for (SentenceAnalysis.Entry entry : sentenceAnalysis) {
			System.out.println("Word = " + entry.input);
			for (WordAnalysis analysis : entry.parses) {
				System.out.println(analysis.formatOflazer());
			}
		}
	}

	public static void main(String[] args) throws Exception {
		
		test_data = new ArrayList<String>();
		openTestDataFile();
		log.info(test_data.toString());
		int a = 1;

		TurkishMorphology morphology = TurkishMorphology.createWithDefaults();
		List<WordAnalysis> results = morphology.analyze("kalemin");
		results.forEach(s -> System.out.println(s.formatLong()));

		// Gets Path to Text file
		String filePath = new ClassPathResource("devlet82.txt").getFile().getAbsolutePath();

		log.info("Load & Vectorize Sentences....");
		// Strip white space before and after for each line
		SentenceIterator iter = new BasicLineIterator(filePath);
		// Split on white spaces in the line to get words
		TokenizerFactory t = new DefaultTokenizerFactory();

		/*
		 * CommonPreprocessor will apply the following regex to each token:
		 * [\d\.:,"'\(\)\[\]|/?!;]+ So, effectively all numbers, punctuation symbols and
		 * some special symbols are stripped off. Additionally it forces lower case for
		 * all tokens.
		 */
		t.setTokenPreProcessor(new CommonPreprocessor());

		// manual creation of VocabCache and WeightLookupTable usually isn't necessary
		// but in this case we'll need them
		VocabCache<VocabWord> cache = new AbstractCache<>();
		WeightLookupTable<VocabWord> table = new InMemoryLookupTable.Builder<VocabWord>().vectorLength(100)
				.useAdaGrad(false).cache(cache).build();

		log.info("Building model....");
		Word2Vec vec = new Word2Vec.Builder().minWordFrequency(10) // TODO: will be 2
				.iterations(1).epochs(1) // TODO: will be 3
				.layerSize(300).seed(42).windowSize(5).iterate(iter).tokenizerFactory(t).lookupTable(table)
				.vocabCache(cache).build();

		log.info("Fitting Word2Vec model....");
		vec.fit();

		log.info("Writing word vectors to text file....");

		// Prints out the closest 10 words to "day". An example on what to do with these
		// Word Vectors.
		log.info("Closest Words:");
		Collection<String> lst = vec.wordsNearestSum("evet", 10);
		log.info("10 Words closest to 'evet': {}", lst);

		lst = vec.wordsNearestSum("sevinç", 10);
		log.info("10 Words closest to 'sevinç': {}", lst);

		// lst = vec.wordsNearestSum("neşe", 10);
		// log.info("10 Words closest to 'neşe': {}", lst);

		log.info("Cosine Similarity (The closer it is to 1, the more similar the net perceives those words to be");
		Double cosineSim = vec.similarity("sevinç", "neşe");
		log.info("Cosine Similarity between 'sevinç' and 'neşe' : " + cosineSim);

		cosineSim = vec.similarity("evet", "öyle");
		log.info("Cosine Similarity between 'evet' and 'öyle' : " + cosineSim);

		// Finding most frequently used words

		List<VocabWord> list = new ArrayList<VocabWord>(cache.vocabWords());
		List<VocabWord> sortedList = new ArrayList<VocabWord>(cache.vocabWords());
		Comparator<VocabWord> comparator = new Comparator<VocabWord>() {
			@Override
			public int compare(VocabWord left, VocabWord right) {
				return cache.wordFrequency(right.getWord()) - cache.wordFrequency(left.getWord());
			}
		};

		Collections.sort((List<VocabWord>) sortedList, comparator);
		System.out.println(cache.numWords());
		System.out.println(cache.totalWordOccurrences());

		for (int i = 0; i < 50; i++) {
			String word = sortedList.get(i).getWord();
			System.out.println(word + " " + cache.wordFrequency(word));
		}

	}

	private static void openTestDataFile() throws IOException {

		String line;
		BufferedReader in;
		log.info(new ClassPathResource("123.txt").getFile().getAbsolutePath());
		in = new BufferedReader(new FileReader(new ClassPathResource("123.txt").getFile().getAbsolutePath()));
		line = in.readLine();

		while (line != null) {
			test_data.add(line);
			System.out.println(line);
			line = in.readLine();
		}

		System.out.println(line);

	}
}
