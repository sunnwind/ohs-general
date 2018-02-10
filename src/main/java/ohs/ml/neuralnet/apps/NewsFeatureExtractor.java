package ohs.ml.neuralnet.apps;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.matrix.DenseVector;
import ohs.ml.svm.wrapper.LibLinearWrapper;
import ohs.nlp.ling.types.LDocument;
import ohs.nlp.ling.types.LDocumentCollection;
import ohs.nlp.ling.types.LSentence;
import ohs.nlp.ling.types.LToken;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.generic.SetMap;
import ohs.types.generic.Vocab.SYM;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.UnicodeUtils;

public class NewsFeatureExtractor {

	private Indexer<String> featIdxer = Generics.newIndexer();

	private Map<Integer, Indexer<String>> valIdxerMap = Generics.newHashMap();

	private SetMap<String, String> gzData = Generics.newSetMap();

	private boolean is_training = false;

	private int unk = 0;

	private LibLinearWrapper llw;

	public NewsFeatureExtractor() {
		int idx = featIdxer.getIndex("word");
		valIdxerMap.put(idx, Generics.newIndexer());
		Indexer<String> wordIdxer = valIdxerMap.get(idx);
		wordIdxer.add(SYM.UNK.getText());
	}

	public void addCapitalFeatures() {
		String feat = "caps";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.getIndex(String.format("%s=%s", feat, "nocaps"));
		featValIdxer.getIndex(String.format("%s=%s", feat, "allcaps"));
		featValIdxer.getIndex(String.format("%s=%s", feat, "initcap"));
		featValIdxer.getIndex(String.format("%s=%s", feat, "hascap"));
	}

	public void addCharacterFeatures() {
		String feat = "ch";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());
		Indexer<String> valIdxer = valIdxerMap.get(idx);
		valIdxer.add(String.format("%s=%s", feat, "unk"));
	}

	public void addDigitFeatures() {
		String feat = "digit";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.add(String.format("%s=%s", feat, "nodigits"));
	}

	public void addGazeteerFeatures(String name, Set<String> data) {
		String feat = String.format("%s=%s", "gz", name);
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
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
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.getIndex(String.format("%s=%s", feat, "nopos"));

		for (String pos : poss) {
			featValIdxer.getIndex(String.format("%s=%s", feat, pos));
		}
	}

	public void addPrefixFeatures(Set<String> prefixes) {
		String feat = "pre";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.getIndex(String.format("%s=%s", feat, "nopre"));

		for (String p : prefixes) {
			featValIdxer.getIndex(String.format("%s=%s", feat, p));
		}
	}

	public void addPuctuationFeatures() {
		String feat = "punc";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.getIndex(String.format("%s=%s", feat, "yes"));
		featValIdxer.getIndex(String.format("%s=%s", feat, "no"));
	}

	public void addShapeOneFeatures() {
		String feat = "shape1";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.add(String.format("%s=%s", feat, "noshape"));
	}

	public void addShapeThreeFeatures() {
		String feat = "shape3";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.add(String.format("%s=%s", feat, "noshape"));
	}

	public void addShapeTwoFeatures() {
		String feat = "shape2";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.add(String.format("%s=%s", feat, "noshape"));
	}

	public void addSuffixFeatures(Set<String> suffixes) {
		String feat = "suf";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.getIndex(String.format("%s=%s", feat, "nosuf"));

		for (String suffix : suffixes) {
			featValIdxer.getIndex(String.format("%s=%s", feat, suffix));
		}
	}

	public void addTopicFeatures(LibLinearWrapper llw) {
		this.llw = llw;

		// String[] feats = { "doc-topic", "sent-topic" };
		String[] feats = { "sent-topic" };

		for (int i = 0; i < feats.length; i++) {
			String feat = feats[i];
			int idx = featIdxer.getIndex(feat);
			valIdxerMap.put(idx, Generics.newIndexer());

			Indexer<String> featValIdxer = valIdxerMap.get(idx);

			for (String val : llw.getLabelIndexer().getObjects()) {
				featValIdxer.getIndex(String.format("%s=%s", feat, val));
			}
		}
	}

	public void addWordSectionFeatures() {
		String feat = "word_sec";

		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);

		String[] vals = { "in_title", "in_body", "in_both" };

		for (String val : vals) {
			featValIdxer.getIndex(String.format("%s=%s", feat, val));
		}
	}

	public void addWordLocationFeatures1() {
		String feat = "word_loc";

		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);

		String[] vals = { "single", "multiple" };

		for (String val : vals) {
			featValIdxer.getIndex(String.format("%s=%s", feat, val));
		}
	}

	public void extract(LDocumentCollection c) {
		for (LDocument d : c) {
			extract(d);
		}
	}

	public void extract(LDocument d) {
		for (LSentence s : d) {
			extract(s);
		}
		extractTopicFeatures(d);
		extractWordSectionFeatures(d);
	}

	public void extract(LSentence s) {
		for (int i = 0; i < s.size(); i++) {
			LToken t = s.get(i);
			extractTokenFeatures(t);
		}
		extractChunkFeatures(s);
		extractSentenceTopicFeatures(s);
	}

	private void extractChunkFeatures(LSentence s) {
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
							// System.out.printf("match: [%s]\n", sub);
							boolean check = true;
							check = false;
							for (int u = i; u < j; u++) {
								feat_flags[u][k] = true;
							}
						}
					}
				}
			}

			for (int i = 0; i < s.size(); i++) {
				LToken t = s.get(i);
				DenseVector F = t.getFeatureVector();
				boolean[] flags = feat_flags[i];

				for (int j = 0; j < flags.length; j++) {
					String feat = feats.get(j);
					String val = feat + "_no";

					if (flags[j]) {
						val = feat + "_yes";
					}

					int feat_idx = featIdxer.indexOf(feat);
					Indexer<String> valIdxer = valIdxerMap.get(feat_idx);
					int val_idx = valIdxer.indexOf(val);
					F.set(feat_idx, val_idx);
				}
			}
		}
	}

	private void extractTopicFeatures(LDocument d) {
		String feat = "doc-topic";

		if (llw == null || !featIdxer.contains(feat)) {
			return;
		}

		Counter<String> c = llw.score(d.getCounter(0));

		String topic = c.argMax();
		String val = String.format("%s=%s", feat, topic);

		int f_idx = featIdxer.indexOf(feat);
		int v_idx = valIdxerMap.get(f_idx).indexOf(val);

		for (LToken t : d.getTokens()) {
			DenseVector F = t.getFeatureVector();
			F.add(f_idx, v_idx);
		}
	}

	private void extractSentenceTopicFeatures(LSentence s) {
		String feat = "sent-topic";

		if (llw == null || !featIdxer.contains(feat)) {
			return;
		}

		Counter<String> c = llw.score(s.getCounter(0));

		String topic = c.argMax();

		String val = String.format("%s=%s", feat, topic);
		int f_idx = featIdxer.indexOf(feat);

		Indexer<String> valIdxer = valIdxerMap.get(f_idx);
		int v_idx = valIdxer.indexOf(val);

		for (LToken t : s) {
			DenseVector F = t.getFeatureVector();
			F.add(f_idx, v_idx);
		}
	}

	private void extractWordSectionFeatures(LDocument d) {
		String feat = "word_sec";

		if (!featIdxer.contains(feat)) {
			return;
		}

		CounterMap<String, String> cm = Generics.newCounterMap();

		for (int i = 0; i < d.size(); i++) {
			LSentence s = d.get(i);

			for (int j = 0; j < s.size(); j++) {
				LToken t = s.get(j);
				String word = t.getString(0);
				String pos = t.getString(1);

				if (word.equals(LSentence.STARTING) || word.equals(LSentence.ENDING)) {
					continue;
				}

				if (pos.startsWith("J") || pos.startsWith("X") || pos.startsWith("E")) {
					continue;
				}

				if (pos.startsWith("SSC") || pos.startsWith("SC") || pos.startsWith("SSO") || pos.startsWith("SY")
						|| pos.startsWith("SF")) {
					continue;
				}

				String sec = "body";

				if (i == 0) {
					sec = "title";
				}
				cm.incrementCount(word, sec, 1);
			}
		}

		int feat_idx = featIdxer.indexOf(feat);
		Indexer<String> valIdxer = valIdxerMap.get(feat_idx);

		for (int i = 0; i < d.size(); i++) {
			LSentence s = d.get(i);

			for (LToken t : s) {
				DenseVector F = t.getFeatureVector();
				String word = t.getString(0);
				String pos = t.getString(1);
				int val_idx = 0;

				Counter<String> c = cm.getCounter(word);

				if (c.size() == 2) {
					val_idx = valIdxer.indexOf(String.format("%s=%s", feat, "in_both"));
				} else {
					if (c.containsKey("in_title")) {
						val_idx = valIdxer.indexOf(String.format("%s=%s", feat, "in_title"));
					} else if (c.containsKey("in_body")) {
						val_idx = valIdxer.indexOf(String.format("%s=%s", feat, "in_body"));
					}
				}

				F.add(feat_idx, val_idx);
			}
		}
	}

	private void extractTokenFeatures(LToken t) {
		String word = t.getString(0);
		String shape = getShape(word);

		DenseVector F = t.getFeatureVector();

		if (F == null) {
			int size = featIdxer.size();
			if (featIdxer.contains("ch")) {
				size += word.length();
			}
			F = new DenseVector(size);
			t.setFeatureVector(F);
		}

		for (int feat_idx = 0; feat_idx < featIdxer.size(); feat_idx++) {
			String feat = featIdxer.getObject(feat_idx);
			Indexer<String> valIdxer = valIdxerMap.get(feat_idx);

			if (feat.equals("word")) {
				String nw = StrUtils.normalizeNumbers(word.toLowerCase());
				int w = is_training ? valIdxer.getIndex(nw) : valIdxer.indexOf(nw, unk);
				F.set(feat_idx, w);

			} else if (feat.equals("pos")) {
				int val_idx = valIdxer.indexOf(String.format("%s=%s", feat, "nopos"));
				if (word.equals(LSentence.STARTING) || word.equals(LSentence.ENDING)) {

				} else {
					String pos = t.getString(1);
					String val = String.format("%s=%s", feat, pos);
					val_idx = valIdxer.indexOf(val);

					if (val_idx < 0) {
						val_idx = valIdxer.indexOf(String.format("%s=%s", feat, "nopos"));
					}
				}

				F.set(feat_idx, val_idx);
			} else if (feat.equals("suf")) {
				String lWord = word.toLowerCase();
				List<Integer> feat_idxs = Generics.newArrayList();

				int s = 1;
				int e = 3;

				for (int j = s; j < lWord.length() && j < e; j++) {
					String suffix = lWord.substring(lWord.length() - j - 1, lWord.length());
					String val = String.format("%s=%s", feat, suffix);
					int val_idx = valIdxer.indexOf(val);

					if (val_idx < 0) {
						continue;
					}

					feat_idxs.add(val_idx);
				}

				int val_idx = valIdxer.indexOf("suf=nosuf");

				if (feat_idxs.size() > 0) {
					val_idx = feat_idxs.get(feat_idxs.size() - 1);
				}

				F.set(feat_idx, val_idx);
			} else if (feat.equals("pre")) {
				String lWord = word.toLowerCase();
				int val_idx = valIdxer.indexOf(String.format("%s=%s", feat, "nopre"));

				if (lWord.length() > 3) {
					String val = String.format("%s=%s", feat, lWord.substring(0, 3));
					val_idx = valIdxer.indexOf(val);

					if (val_idx < 0) {
						val_idx = valIdxer.indexOf(String.format("%s=%s", feat, "nopre"));
					}
				}

				F.set(feat_idx, val_idx);
			} else if (feat.equals("punc")) {
				Set<Integer> locs = Generics.newHashSet();

				for (int i = 0; i < shape.length(); i++) {
					char ch = shape.charAt(i);
					if (ch == 'A' || ch == 'a' || ch == '0') {

					} else {
						locs.add(i);
					}
				}
				String val = String.format("%s=%s", feat, locs.size() > 0 ? "yes" : "no");
				int val_idx = valIdxer.indexOf(val);

				F.set(feat_idx, val_idx);
			} else if (feat.equals("caps")) {
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

				F.set(feat_idx, valIdxer.indexOf(String.format("%s=%s", feat, val)));
			} else if (feat.equals("shape1")) {
				String val = String.format("%s=%s", feat, shape);
				int val_idx = is_training ? valIdxer.getIndex(val) : valIdxer.indexOf(val);

				if (val_idx < 0) {
					val_idx = valIdxer.indexOf(String.format("%s=%s", feat, "noshape"));
				}

				F.set(feat_idx, val_idx);
			} else if (feat.equals("shape2")) {
				String shape2 = shape.replaceAll("a+", "a~").replaceAll("A{2,}", "A~A").replaceAll("K+", "K~");
				String val = String.format("%s=%s", feat, shape2);
				int val_idx = is_training ? valIdxer.getIndex(val) : valIdxer.indexOf(val);

				if (val_idx < 0) {
					val_idx = valIdxer.indexOf(String.format("%s=%s", feat, "noshape"));
				}

				F.set(feat_idx, val_idx);
			} else if (feat.equals("shape3")) {
				String suffix = "";
				int suffix_len = 2;

				if (word.length() > 3) {
					int loc = word.length() - suffix_len;
					shape = getShape(word.substring(0, loc));
					suffix = word.substring(loc).toLowerCase();
				}

				String shape2 = shape.replaceAll("a+", "a~").replaceAll("A{2,}", "A~A").replaceAll("K+", "K~");
				String val = String.format("%s=%s", feat, shape2);
				if (suffix.length() > 0) {
					val += "|" + suffix;
				}

				int val_idx = is_training ? valIdxer.getIndex(val) : valIdxer.indexOf(val);

				if (val_idx < 0) {
					val_idx = valIdxer.indexOf(String.format("%s=%s", feat, "noshape"));
				}

				F.set(feat_idx, val_idx);
			} else if (feat.equals("ch")) {
				for (int i = 0; i < word.length(); i++) {
					String ch = word.charAt(i) + "";
					String val = String.format("%s=%s", feat, ch);
					int val_idx = is_training ? valIdxer.getIndex(val) : valIdxer.indexOf(val, 0);
					F.add(featIdxer.size() - 1 + i, val_idx);
				}
			}
		}
	}

	public Indexer<String> getFeatureIndexer() {
		return featIdxer;
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
			} else if (UnicodeUtils.isKorean(c)) {
				sb.append("K");
			} else {
				sb.append(c);
			}
		}
		return sb.toString();
	}

	public List<Indexer<String>> getValueIndexers() {
		List<Indexer<String>> ret = Generics.newArrayList(featIdxer.size());
		for (int i = 0; i < featIdxer.size(); i++) {
			ret.add(valIdxerMap.get(i));
		}
		return ret;
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		featIdxer = FileUtils.readStringIndexer(ois);

		valIdxerMap = Generics.newHashMap(featIdxer.size());

		for (int i = 0; i < featIdxer.size(); i++) {
			valIdxerMap.put(i, FileUtils.readStringIndexer(ois));
		}

		gzData = FileUtils.readStringSetMap(ois);
		is_training = ois.readBoolean();
		unk = ois.readInt();
	}

	public void readObject(String fileName) throws Exception {
		System.out.printf("read at [%s]\n", fileName);
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public void setIsTraining(boolean is_training) {
		this.is_training = is_training;
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		FileUtils.writeStringIndexer(oos, featIdxer);

		for (int i = 0; i < featIdxer.size(); i++) {
			FileUtils.writeStringIndexer(oos, valIdxerMap.get(i));
		}

		FileUtils.writeStringSetMap(oos, gzData);
		oos.writeBoolean(is_training);
		oos.writeInt(unk);
	}

	public void writeObject(String fileName) throws Exception {
		System.out.printf("write at [%s]\n", fileName);
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName, false);
		writeObject(oos);
		oos.flush();
		oos.close();
	}
}
