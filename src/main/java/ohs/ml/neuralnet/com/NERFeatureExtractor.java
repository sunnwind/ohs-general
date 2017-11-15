package ohs.ml.neuralnet.com;

import java.util.List;
import java.util.Set;

import org.apache.xmlbeans.impl.xb.xsdschema.Attribute.Use;

import ohs.matrix.DenseVector;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MToken;
import ohs.types.generic.Indexer;
import ohs.types.generic.ListMap;
import ohs.types.generic.SetMap;
import ohs.types.generic.Vocab;
import ohs.types.generic.Vocab.SYM;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class NERFeatureExtractor {

	private Indexer<String> featIdxer = Generics.newIndexer();

	private Indexer<String> featValIdxer = Generics.newIndexer();

	private Indexer<String> labelIdxer = Generics.newIndexer();

	private SetMap<String, String> gzData = Generics.newSetMap();

	private ListMap<String, Integer> lm = Generics.newListMap();

	private Vocab vocab = new Vocab();

	public boolean add_unkwon_words = true;

	private int unk = 0;

	public NERFeatureExtractor() {
		vocab.add(SYM.UNK.getText());

		featValIdxer.add("word");

		featIdxer.add("word");
	}

	private boolean add_pos_feats = false;

	private boolean add_punc_feats = false;

	public void addCapitalFeatures() {
		String feat = "caps";
		featIdxer.add(feat);

		lm.put(feat, featValIdxer.getIndex(String.format("%s=%s", feat, "nocaps")));
		lm.put(feat, featValIdxer.getIndex(String.format("%s=%s", feat, "allcaps")));
		lm.put(feat, featValIdxer.getIndex(String.format("%s=%s", feat, "initcap")));
		lm.put(feat, featValIdxer.getIndex(String.format("%s=%s", feat, "hascap")));

	}

	public void addGazeteerFeatures(String name, Set<String> data) {
		String feat = String.format("%s=%s", "gz", name);
		featIdxer.add(feat);

		lm.put(feat, featValIdxer.getIndex(String.format("%s=%s_%s", "gz", name, "yes")));
		lm.put(feat, featValIdxer.getIndex(String.format("%s=%s_%s", "gz", name, "no")));

		Set<String> newData = Generics.newHashSet(data.size());

		for (String s : data) {
			newData.add(s.replaceAll("[\\s\u2029]+", "").trim());
		}
		gzData.put(feat, newData);
	}

	public void addSuffixFeatures(Set<String> suffixes) {
		String feat = "suf";
		featIdxer.add(feat);

		lm.put(feat, featValIdxer.getIndex(String.format("%s=%s", feat, "<nosuf>")));

		for (String suffix : suffixes) {
			lm.put(feat, featValIdxer.getIndex(String.format("%s=%s", feat, suffix)));
		}
	}

	private String[] PUNCUTATIONS = { "+", "-", ",", ".", "`", "?", "$" };

	public void addPuctuationFeatures() {
		add_punc_feats = true;

		featIdxer.add(String.format("punc=%s", "<nopunc>"));

		for (String p : PUNCUTATIONS) {
			featIdxer.add(String.format("punc=%s", p));
			featValIdxer.add(String.format("punc=%s", p));
		}
	}

	public void addPosFeatures(Set<String> poss) {
		String feat = "pos";
		featIdxer.add(feat);

		lm.put(feat, featValIdxer.getIndex(String.format("%s=%s", feat, "nopos")));

		for (String pos : poss) {
			lm.put(feat, featValIdxer.getIndex(String.format("%s=%s", feat, pos)));
		}

		add_pos_feats = true;
	}

	public void extract(MDocument d) {

	}

	public void extract(MSentence s) {
		for (int i = 0; i < s.size(); i++) {
			MToken t = s.get(i);
			extractTokenFeatures(t);
		}
		extractChunkFeatures(s);
	}

	public void extractChunkFeatures(MSentence s) {
		if (gzData.size() > 0) {
			List<String> words = s.getTokenStrings(0);

			for (int i = 0; i < words.size(); i++) {
				words.set(i, words.get(i).toLowerCase());
			}

			List<String> feats = Generics.newArrayList(gzData.keySet());

			boolean[][] feat_flags = new boolean[words.size()][feats.size()];

			int win_size = 5;

			for (int i = 0; i < s.size(); i++) {
				for (int j = i + 1; j < Math.min(s.size(), i + win_size); j++) {
					String sub = StrUtils.join("", words, i, j);

					for (int k = 0; k < feats.size(); k++) {
						String feat = feats.get(k);
						Set<String> gz = gzData.get(feat);

						if (gz.contains(sub)) {
							for (int u = i; u < j; u++) {
								feat_flags[u][k] = true;
							}
						}
					}
				}
			}

			for (int i = 0; i < s.size(); i++) {
				MToken t = s.get(i);
				DenseVector F = (DenseVector) t.get(t.size() - 1);
				boolean[] flags = feat_flags[i];

				for (int j = 0; j < flags.length; j++) {
					String feat = feats.get(j);
					String val = feat + "_no";

					if (flags[j]) {
						val = feat + "_yes";
					}

					F.set(featIdxer.indexOf(feat), featValIdxer.indexOf(val));
				}
			}
		}
	}

	public void extractTokenFeatures(MToken t) {
		DenseVector F = new DenseVector(featIdxer.size());

		String word = t.getString(0);

		{
			String nWord = StrUtils.normalizeNumbers(word.toLowerCase());
			String val = nWord;
			int idx = add_unkwon_words ? vocab.getIndex(val) : vocab.indexOf(val, unk);
			F.set(featIdxer.indexOf("word"), idx);
		}

		if (featIdxer.contains("pos")) {
			String pos = t.getString(1);
			String val = String.format("pos=%s", pos);
			F.set(featIdxer.indexOf("pos"), add_unkwon_words ? featValIdxer.getIndex(val) : featValIdxer.indexOf(val));
		}

		Set<Integer> caps = Generics.newHashSet(word.length());

		for (int i = 0; i < word.length(); i++) {
			char ch = word.charAt(i);
			if (Character.isUpperCase(ch)) {
				caps.add(i);
			}
		}

		if (featIdxer.contains("caps")) {
			String val = "nocaps";

			if (caps.size() == word.length()) {
				val = "allcaps";
			} else {
				if (caps.contains(0)) {
					val = "initcap";
				} else {
					val = "hascap";
				}
			}
			F.set(featIdxer.indexOf("caps"), featValIdxer.indexOf(String.format("caps=%s", val)));
		}

		if (featIdxer.contains("suf")) {
			String lWord = word.toLowerCase();
			List<Integer> feat_idxs = Generics.newArrayList();

			for (int j = 0; j < lWord.length() && j < 3; j++) {
				String suffix = lWord.substring(lWord.length() - j - 1, lWord.length());
				String val = String.format("suf=%s", suffix);
				int feat_idx = featValIdxer.indexOf(val);

				if (feat_idx < 0) {
					continue;
				}

				feat_idxs.add(feat_idx);
			}

			int feat_idx = featValIdxer.indexOf("suf=<nosuf>");

			if (feat_idxs.size() > 0) {
				feat_idx = feat_idxs.get(feat_idxs.size() - 1);
			}

			F.set(featIdxer.indexOf("suf"), feat_idx);
		}

		if (add_punc_feats) {
			for (String p : PUNCUTATIONS) {
				if (word.contains(p)) {
					String feat = String.format("punc=%s", p);
					F.set(featIdxer.indexOf(feat), featValIdxer.indexOf(feat));
				}

			}
		}

		t.add(F);
	}

	public Indexer<String> getFeatureIndexer() {
		return featIdxer;
	}

	public Indexer<String> getFeatureValueIndexer() {
		return featValIdxer;
	}

	public Vocab getVocab() {
		return vocab;
	}

	public void setAddUnkwonWords(boolean add_unkwon_words) {
		this.add_unkwon_words = add_unkwon_words;
	}

	public void setVocab(Vocab vocab) {
		this.vocab = vocab;
	}
}
