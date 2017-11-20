package ohs.ir.medical.general;

import ohs.math.ArrayMath;
import ohs.ml.word2vec.Word2VecModel;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;

public class Word2VecSearcher {
	private Indexer<String> wordIndexer;

	private int layerSize;

	private double[][] vectors;

	public Word2VecSearcher(Word2VecModel model) {
		wordIndexer = new Indexer<>();
		for (int i = 0; i < model.getWordVocab().getBytes(); i++) {
			wordIndexer.add(model.getWordVocab().getSents(i));
		}
		
		

		System.out.println(model.getWordVocab().getBytes());
		System.out.println(wordIndexer.size());

		this.layerSize = model.getLayerSize();
		this.vectors = model.getVectors();

		for (int i = 0; i < vectors.length; i++) {
			ArrayMath.unitVector(vectors[i], vectors[i]);
		}
	}

	public int getLayerSize() {
		return layerSize;
	}

	public double[] getVector(String word) {
		int w = wordIndexer.indexOf(word);
		double[] ret = null;
		if (w > -1) {
			ret = vectors[w];
		}
		return ret;
	}

	public double[][] getVectors() {
		return vectors;
	}

	public Indexer<String> getWordIndexer() {
		return wordIndexer;
	}

	public Counter<String> search(String word) {
		Counter<String> ret = new Counter<String>();

		int w = wordIndexer.indexOf(word);

		if (w > -1) {
			for (int i = 0; i < vectors.length; i++) {
				if (i == w) {
					continue;
				}
				String word2 = wordIndexer.getObject(i);
				double dot_product = ArrayMath.dotProduct(vectors[w], vectors[i]);
				ret.incrementCount(word2, dot_product);
			}
		}

		return ret;
	}
}
