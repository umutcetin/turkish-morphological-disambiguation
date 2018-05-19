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
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.regex.Pattern;

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
	private static ArrayList<String> output;

	TurkishSentenceAnalyzer sentenceAnalyzer;

	private static Logger log = LoggerFactory.getLogger(Word2VecRawTextExample.class);

	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws Exception {
		
		test_data = new ArrayList<String>();
		openTestDataFile(); //updates test_data

		TurkishMorphology morphology = TurkishMorphology.createWithDefaults();
//		List<WordAnalysis> results = morphology.analyze("yaşama");
//		results.forEach(s -> System.out.println(s.formatOflazer()));
		
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
		Word2Vec vec = new Word2Vec.Builder().minWordFrequency(2) 
				.iterations(1).epochs(1) 
				.layerSize(300).seed(42).windowSize(5).iterate(iter).tokenizerFactory(t).lookupTable(table)
				.vocabCache(cache).build();

		log.info("Fitting Word2Vec model....");
		vec.fit();

		log.info("Writing word vectors to text file....");

		// Prints out the closest 10 words to "day". An example on what to do with these
		// Word Vectors.
//		log.info("Closest Words:");
//		Collection<String> lst = vec.wordsNearestSum("a", 1);
//		log.info("10 Words closest to 'a': {}", lst);

//		lst = vec.wordsNearestSum("yaşama", 10);
//		log.info("10 Words closest to 'yaşama': {}", lst);

		// lst = vec.wordsNearestSum("neşe", 10);
		// log.info("10 Words closest to 'neşe': {}", lst);
//
//		log.info("Cosine Similarity (The closer it is to 1, the more similar the net perceives those words to be");
//		Double cosineSim = vec.similarity("sevinç", "neşe");
//		log.info("Cosine Similarity between 'sevinç' and 'neşe' : " + cosineSim);
//
//		cosineSim = vec.similarity("evet", "öyle");
//		log.info("Cosine Similarity between 'evet' and 'öyle' : " + cosineSim);

		
		// Finding most frequently used words- this code was used once

//		List<VocabWord> list = new ArrayList<VocabWord>(cache.vocabWords());
//		List<VocabWord> sortedList = new ArrayList<VocabWord>(cache.vocabWords());
//		Comparator<VocabWord> comparator = new Comparator<VocabWord>() {
//			@Override
//			public int compare(VocabWord left, VocabWord right) {
//				return cache.wordFrequency(right.getWord()) - cache.wordFrequency(left.getWord());
//			}
//		};
//
//		Collections.sort((List<VocabWord>) sortedList, comparator);
//		//System.out.println(cache.numWords());
//		//System.out.println(cache.totalWordOccurrences());
//
//		for (int i = 0; i < 50; i++) {
//			String word = sortedList.get(i).getWord();
//			System.out.println(word + " " + cache.wordFrequency(word));
//		}
		
		//words to escape during similar word selection
		String[] escape_words = {"bir", "ve", "de", "da", "için", "ile", "en", "öyle", "gibi", "değil",  
		                         "ne",  "ki",  "ya",  "a",  "b",  "c" };
		
		for (String test_data_line : test_data) {
			//test_data_line = "yaşama yaşa+Verb+Pos^DB+Noun+Inf2+A3sg+Pnon+Nom yaşam+Noun+A3sg+Pnon+Dat yaşa+Verb+Neg+Imp+A2sg";
			String[] parts = test_data_line.split(" ");
			String test_word = parts[0];
			String[] options = Arrays.copyOfRange(parts, 1, parts.length);
			
			List<List<String>> tw_option_features = new ArrayList<List<String>>();
			//get features for each option
			for (String option : options) {
				String[] twp_parts = option.split(Pattern.quote("+"));
				String[] twp_features = Arrays.copyOfRange(twp_parts, 1, twp_parts.length);
				
				List<String> twp_features_list = new ArrayList<String>();
				for (String feature : twp_features) {
					twp_features_list.add(feature);
				}
				
				tw_option_features.add(twp_features_list);
			}
			
			
			//get closest words to current test word 
			Collection<String> similar_words = new ArrayList<String>();
			try {
				similar_words = vec.wordsNearestSum(test_word, 5);
			}
			catch(Exception e) {
				System.out.println(e.toString());
			}
			finally {
				
			}
			
			
			//remove nulls from this list
			for (String sw : similar_words) {
				if(sw == null)
					similar_words.remove(sw);
			}
			
			//remove escape words from this list
			for (String escape_word : escape_words) {
				similar_words.remove(escape_word);
			}
			
			//get morphological parses for each similar word
			for (String similar_word : similar_words) {
				List<String> similar_word_parses = new ArrayList<String>();
				List<WordAnalysis> results = morphology.analyze(similar_word);
				results.forEach(s -> System.out.println(s.formatOflazer()));
				results.forEach(s -> similar_word_parses.add(s.formatOflazer()));
				
				List<String> feature_pool = new ArrayList<String>();
				
				//remove first elements from parses since they are word roots, not features
				//collect features from similar word parses
				for (String similar_word_parse : similar_word_parses) {
					String[] swp_parts = similar_word_parse.split(Pattern.quote("+"));
					String[] swp_features = Arrays.copyOfRange(swp_parts, 1, swp_parts.length);
					
					for (String feature : swp_features) {
						feature_pool.add(feature);
					}
				}

				
				//sort feature pool by frequency
				Map<String, Integer> map = new HashMap<String, Integer>();

				for (String temp : feature_pool) {
					Integer count = map.get(temp);
					map.put(temp, (count == null) ? 1 : count + 1);
				}
				
				
				//sorted map
				Object[] sorted_map = map.entrySet().toArray();
				Arrays.sort(sorted_map, new Comparator<Object>() {
				    public int compare(Object o1, Object o2) {
				        return ((Map.Entry<String, Integer>) o2).getValue()
				                   .compareTo(((Map.Entry<String, Integer>) o1).getValue());
				    }
				});

				//select features for the word option output
				List<String> output_features = new ArrayList<String>();
				
				for(int i = 0; i < sorted_map.length; i++) {
					Map.Entry<String, Integer> o =(Entry<String, Integer>) sorted_map[i];
					if(o.getValue() > 1) {
						output_features.add(o.getKey());
					}
				}
				
				
				int max_similarity = 0;
				int max_index = 0;
				//compare output features with options
				for(int i=0; i<tw_option_features.size(); i++) {
					//intersection of features
					List<String> fl = tw_option_features.get(i);
					fl.retainAll(output_features);
					int similarity = fl.size();
					
					if(similarity>max_similarity) {
						max_similarity = similarity;
						max_index = i;
					}
				}
				
				//add selection to output
				output.add(test_word +" "+ tw_option_features.get(max_index));
				
				
			}
			
		}

	}
	
	private static void openTestDataFile() throws IOException {

		String line;
		BufferedReader in;
		//log.info(new ClassPathResource("test_data.txt").getFile().getAbsolutePath());
		in = new BufferedReader(new FileReader(new ClassPathResource("test_data.txt").getFile().getAbsolutePath()));
		line = in.readLine();

		while (line != null) {
			test_data.add(line);
			//System.out.println(line);
			line = in.readLine();
		}
		in.close();

	}
}
