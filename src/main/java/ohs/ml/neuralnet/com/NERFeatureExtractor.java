package ohs.ml.neuralnet.com;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.matrix.DenseVector;
import ohs.nlp.ling.types.MCollection;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MToken;
import ohs.types.generic.Indexer;
import ohs.types.generic.SetMap;
import ohs.types.generic.Vocab;
import ohs.types.generic.Vocab.SYM;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class NERFeatureExtractor {

	private Indexer<String> featIdxer = Generics.newIndexer();

	private Indexer<String> featValIdxer = Generics.newIndexer();

	private SetMap<String, String> gzData = Generics.newSetMap();

	private Vocab vocab = new Vocab();

	public boolean is_training = true;

	private int unk = 0;

	public NERFeatureExtractor() {
		vocab.add(SYM.UNK.getText());
		featIdxer.add("word");
	}

	public void addCapitalFeatures() {
		String feat = "caps";
		featIdxer.add(feat);

		featValIdxer.getIndex(String.format("%s=%s", feat, "nocaps"));
		featValIdxer.getIndex(String.format("%s=%s", feat, "allcaps"));
		featValIdxer.getIndex(String.format("%s=%s", feat, "initcap"));
		featValIdxer.getIndex(String.format("%s=%s", feat, "hascap"));
	}

	public void addDigitFeatures() {
		String feat = "digit";
		featIdxer.add(feat);
		featValIdxer.add(String.format("%s=%s", feat, "nodigits"));
	}

	public void addGazeteerFeatures(String name, Set<String> data) {
		String feat = String.format("%s=%s", "gz", name);
		featIdxer.add(feat);

		featValIdxer.getIndex(String.format("%s=%s_%s", "gz", name, "yes"));
		featValIdxer.getIndex(String.format("%s=%s_%s", "gz", name, "no"));

		Set<String> newData = Generics.newHashSet(data.size());

		for (String s : data) {
			newData.add(s.replaceAll("[\\s\u2029]+", "").trim());
		}
		gzData.put(feat, newData);
	}

	public void addPosFeatures(Set<String> poss) {
		String feat = "pos";
		featIdxer.add(feat);

		featValIdxer.getIndex(String.format("%s=%s", feat, "nopos"));

		for (String pos : poss) {
			featValIdxer.getIndex(String.format("%s=%s", feat, pos));
		}
	}

	public void addPrefixFeatures(Set<String> prefixes) {
		String feat = "pre";
		featIdxer.add(feat);
		featValIdxer.getIndex(String.format("%s=%s", feat, "nopre"));

		for (String p : prefixes) {
			featValIdxer.getIndex(String.format("%s=%s", feat, p));
		}
	}

	public void addPuctuationFeatures() {
		String feat = "punc";
		featIdxer.add(String.format("punc"));
		featValIdxer.getIndex(String.format("%s=%s", feat, "yes"));
		featValIdxer.getIndex(String.format("%s=%s", feat, "no"));
	}

	public void addShapeOneFeatures() {
		String feat = "shape1";
		featIdxer.add(feat);
		featValIdxer.add(String.format("%s=%s", feat, "noshape"));
	}

	public void addShapeTwoFeatures() {
		String feat = "shape2";
		featIdxer.add(feat);
		featValIdxer.add(String.format("%s=%s", feat, "noshape"));
	}

	public void addSuffixFeatures(Set<String> suffixes) {
		String feat = "suf";
		featIdxer.add(feat);
		featValIdxer.getIndex(String.format("%s=%s", feat, "nosuf"));

		for (String suffix : suffixes) {
			featValIdxer.getIndex(String.format("%s=%s", feat, suffix));
		}
	}

	public void extract(MCollection c) {
		for (MDocument d : c) {
			extract(d);
		}
	}

	public void extract(MDocument d) {
		for (MSentence s : d) {
			extract(s);
		}
	}

	public void extract(MSentence s) {
		for (int i = 0; i < s.size(); i++) {
			MToken t = s.get(i);
			extractTokenFeatures(t);
		}
		extractChunkFeatures(s);
	}

	private void extractChunkFeatures(MSentence s) {
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

	private void extractTokenFeatures(MToken t) {
		DenseVector F = new DenseVector(featIdxer.size());

		String word = t.getString(0);

		{
			String nw = StrUtils.normalizeNumbers(word.toLowerCase());
			int w = is_training ? vocab.getIndex(nw) : vocab.indexOf(nw, unk);
			F.set(featIdxer.indexOf("word"), w);
		}

		if (featIdxer.contains("pos")) {
			String pos = t.getString(1);
			String val = String.format("pos=%s", pos);
			int val_idx = featValIdxer.indexOf(val);

			if (val_idx < 0) {
				val_idx = featValIdxer.indexOf(String.format("pos=%s", "nopos"));
			}
			F.set(featIdxer.indexOf("pos"), val_idx);
		}

		if (featIdxer.contains("suf")) {
			String lWord = word.toLowerCase();
			List<Integer> feat_idxs = Generics.newArrayList();

			int s = 1;
			int e = 3;
			String feat = "suf";

			for (int j = s; j < lWord.length() && j < e; j++) {
				String suffix = lWord.substring(lWord.length() - j - 1, lWord.length());
				String val = String.format("%s=%s", feat, suffix);
				int val_idx = featValIdxer.indexOf(val);

				if (val_idx < 0) {
					continue;
				}

				feat_idxs.add(val_idx);
			}

			int val_idx = featValIdxer.indexOf("suf=nosuf");

			if (feat_idxs.size() > 0) {
				val_idx = feat_idxs.get(feat_idxs.size() - 1);
			}

			F.set(featIdxer.indexOf(feat), val_idx);
		}

		if (featIdxer.contains("pre")) {
			String lWord = word.toLowerCase();
			String feat = "pre";

			int val_idx = featValIdxer.indexOf(String.format("%s=%s", "pre", "nopre"));

			if (lWord.length() > 3) {
				String val = String.format("%s=%s", "pre", lWord.substring(0, 3));
				val_idx = featValIdxer.indexOf(val);

				if (val_idx < 0) {
					val_idx = featValIdxer.indexOf(String.format("%s=%s", "pre", "nopre"));
				}
			}

			F.set(featIdxer.indexOf(feat), val_idx);
		}

		String shape = getShape(word);

		if (featIdxer.contains("punc")) {
			String feat = "punc";
			Set<Integer> locs = Generics.newHashSet();

			for (int i = 0; i < shape.length(); i++) {
				char ch = shape.charAt(i);
				if (ch == 'A' || ch == 'a' || ch == '0') {

				} else {
					locs.add(i);
				}
			}
			String val = String.format("%s=%s", feat, locs.size() > 0 ? "yes" : "no");
			F.set(featIdxer.indexOf(feat), featValIdxer.indexOf(val));
		}

		if (featIdxer.contains("caps")) {
			Set<Integer> caps = Generics.newHashSet(shape.length());
			for (int i = 0; i < shape.length(); i++) {
				if (shape.charAt(i) == 'A') {
					caps.add(i);
				}
			}

			String val = "nocaps";

			if (caps.size() == shape.length()) {
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

		if (featIdxer.contains("shape1")) {
			String feat = "shape1";
			String val = String.format("%s=%s", feat, shape);
			int val_idx = is_training ? featValIdxer.getIndex(val) : featValIdxer.indexOf(val);

			if (val_idx < 0) {
				val_idx = featValIdxer.indexOf(String.format("%s=%s", feat, "noshape"));
			}

			F.set(featIdxer.indexOf(feat), val_idx);
		}

		if (featIdxer.contains("shape2")) {
			String feat = "shape2";
			String val = String.format("%s=%s", feat, shape.replaceAll("a+", "a~").replaceAll("A{2,}", "A~A"));
			int val_idx = is_training ? featValIdxer.getIndex(val) : featValIdxer.indexOf(val);

			if (val_idx < 0) {
				val_idx = featValIdxer.indexOf(String.format("%s=%s", feat, "noshape"));
			}

			F.set(featIdxer.indexOf(feat), val_idx);
		}

		t.add(F);
	}

	public Indexer<String> getFeatureIndexer() {
		return featIdxer;
	}

	public Indexer<String> getFeatureValueIndexer() {
		return featValIdxer;
	}

	private String getShape(String word) {
		StringBuffer sb = new StringBuffer();
		for (char c : word.toCharArray()) {
			if (Character.isDigit(c)) {
				sb.append("0");
			} else if (Character.isUpperCase(c)) {
				sb.append("A");
			} else if (Character.isLowerCase(c)) {
				sb.append("a");
			} else {
				sb.append(c);
			}
		}
		return sb.toString();
	}

	public Vocab getVocab() {
		return vocab;
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		featIdxer = FileUtils.readStringIndexer(ois);
		featValIdxer = FileUtils.readStringIndexer(ois);
		gzData = FileUtils.readStringSetMap(ois);
		vocab = new Vocab(ois);
		is_training = ois.readBoolean();
		unk = ois.readInt();
	}

	public void readObject(String fileName) throws Exception {
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public void setIsTraining(boolean is_training) {
		this.is_training = is_training;
	}

	public void setVocab(Vocab vocab) {
		this.vocab = vocab;
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		FileUtils.writeStringIndexer(oos, featIdxer);
		FileUtils.writeStringIndexer(oos, featValIdxer);
		FileUtils.writeStringSetMap(oos, gzData);
		vocab.writeObject(oos);
		oos.writeBoolean(is_training);
		oos.writeInt(unk);
	}

	public void writeObject(String fileName) throws Exception {
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName, false);
		writeObject(oos);
		oos.flush();
		oos.close();
	}
}
