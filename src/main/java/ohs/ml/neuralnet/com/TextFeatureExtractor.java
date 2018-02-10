package ohs.ml.neuralnet.com;

import java.io.File;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.matrix.DenseVector;
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

public class TextFeatureExtractor {

	private static final String cons = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ";

	private static final String vowels = "aeiouAEIOU";

	public static Map<String, Boolean> DEFAULT_SETTING;

	public static TextFeatureExtractor getTextFeatureExtractor(LDocumentCollection C,
			Map<String, String> binaryValueFiles, Set<String> prefixes, Set<String> suffixes) throws Exception {

		TextFeatureExtractor ext = new TextFeatureExtractor();
		Set<String> poss = null;

		if (LToken.INDEXER.contains("pos")) {
			poss = Generics.newHashSet(C.getTokens().getTokenStrings(1));
		}

		prefixes = Generics.newHashSet();

		if (prefixes == null) {
			Counter<String> c = Generics.newCounter();
			for (String word : C.getTokens().getTokenStrings(0)) {
				if (word.length() > 3) {
					String p = word.substring(0, 3);
					p = p.toLowerCase();

					Set<Integer> l = Generics.newHashSet();

					for (int i = 0; i < p.length(); i++) {
						if (!Character.isDigit(p.charAt(i))) {
							l.add(i);
						}
					}

					if (l.size() == p.length()) {
						c.incrementCount(p, 1);
					}
				}
			}

			c.pruneKeysBelowThreshold(5);
			prefixes = c.keySet();
		}

		if (suffixes == null) {
			Counter<String> c = Generics.newCounter();
			for (String word : C.getTokens().getTokenStrings(0)) {
				if (word.length() > 3) {
					String p = word.substring(word.length() - 3, word.length());
					p = p.toLowerCase();

					Set<Integer> l = Generics.newHashSet();

					for (int i = 0; i < p.length(); i++) {
						if (!Character.isDigit(p.charAt(i))) {
							l.add(i);
						}
					}

					if (l.size() == p.length()) {
						c.incrementCount(p, 1);
					}
				}
			}

			c.pruneKeysBelowThreshold(5);
			suffixes = c.keySet();
		}

		if (poss != null) {
			ext.addPosFeatures(poss);
		}

		ext.addCapitalFeatures();
		ext.addShapeOneFeatures();
		ext.addShapeTwoFeatures();
		ext.addShapeThreeFeatures();
		ext.addVowelConsonantFeatures();
		ext.addSuffixFeatures(suffixes);
		ext.addPrefixFeatures(prefixes);

		if (binaryValueFiles != null) {
			for (String feat : binaryValueFiles.keySet()) {
				String fileName = binaryValueFiles.get(feat);
				Set<String> dict = FileUtils.readStringHashSetFromText(fileName);
				ext.addBinaryFeatures(feat, dict);
			}
		}

		// {
		// Set<String> s1 = Generics.newHashSet();
		// Set<String> s2 = Generics.newHashSet();
		//
		// CounterMap<String, String> cm = FileUtils
		// .readStringCounterMapFromText("../../data/medical_ir/phrs/wiki_phrs.txt",
		// false);
		//
		// for (String title : cm.keySet()) {
		// if (cm.containsKey("wkt")) {
		// s1.add(title.toLowerCase());
		// } else {
		// s2.add(title.toLowerCase());
		// }
		// }
		//
		// ext.addBinaryFeatures("wkt", s1);
		// ext.addBinaryFeatures("wkl", s2);
		// }

		return ext;
	}

	public static boolean isConsanant(char c) {
		return cons.contains(c + "");
	}

	public static boolean isVowel(char c) {

		return vowels.contains(c + "");
	}

	{
		DEFAULT_SETTING = Generics.newTreeMap();

		DEFAULT_SETTING.put("word", true);
		DEFAULT_SETTING.put("pos", true);
		DEFAULT_SETTING.put("digit", true);
		DEFAULT_SETTING.put("pre", true);
		DEFAULT_SETTING.put("punc", true);
		DEFAULT_SETTING.put("suf", true);
		DEFAULT_SETTING.put("shape1", true);
		DEFAULT_SETTING.put("shape2", true);
		DEFAULT_SETTING.put("shape3", true);
		DEFAULT_SETTING.put("vc", true);
	}

	protected Indexer<String> featIdxer = Generics.newIndexer();

	protected Map<Integer, Indexer<String>> valIdxerMap = Generics.newHashMap();

	protected SetMap<String, String> gzData = Generics.newSetMap();

	protected boolean is_training = true;

	protected int unk = 0;

	public TextFeatureExtractor() {
		int idx = featIdxer.getIndex("word");
		valIdxerMap.put(idx, Generics.newIndexer());
		Indexer<String> wordIdxer = valIdxerMap.get(idx);
		wordIdxer.add(SYM.UNK.getText());
		wordIdxer.add(LSentence.STARTING);
		wordIdxer.add(LSentence.ENDING);
	}

	public void addCapitalFeatures() {
		String feat = "caps";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.add(String.format("%s=%s", feat, "ignore"));
		featValIdxer.add(String.format("%s=%s", feat, "nocaps"));
		featValIdxer.add(String.format("%s=%s", feat, "allcaps"));
		featValIdxer.add(String.format("%s=%s", feat, "initcap"));
		featValIdxer.add(String.format("%s=%s", feat, "hascap"));
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
		featValIdxer.getIndex(String.format("%s=%s", feat, "ignore"));
		featValIdxer.add(String.format("%s=%s", feat, "nodigits"));
	}

	public void addBinaryFeatures(String name, Set<String> data) {
		String feat = String.format("%s=%s", "bin", name);
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.add(String.format("%s=%s_%s", "bin", name, "ignore"));
		featValIdxer.add(String.format("%s=%s_%s", "bin", name, "yes"));
		featValIdxer.add(String.format("%s=%s_%s", "bin", name, "no"));

		Set<String> newData = Generics.newHashSet(data.size());

		for (String s : data) {
			newData.add(s.replaceAll("[\\s\u2029]+", " ").trim());
		}
		gzData.put(feat, newData);
	}

	public void addPosFeatures(Set<String> poss) {
		String feat = "pos";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.add(String.format("%s=%s", feat, "ignore"));
		featValIdxer.add(String.format("%s=%s", feat, "nopos"));

		for (String pos : poss) {
			featValIdxer.add(String.format("%s=%s", feat, pos));
		}
	}

	public void addPrefixFeatures(Set<String> prefixes) {
		String feat = "pre";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.add(String.format("%s=%s", feat, "ignore"));
		featValIdxer.add(String.format("%s=%s", feat, "nopre"));

		for (String p : prefixes) {
			featValIdxer.add(String.format("%s=%s", feat, p));
		}
	}

	public void addPuctuationFeatures() {
		String feat = "punc";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.add(String.format("%s=%s", feat, "ignore"));
		featValIdxer.add(String.format("%s=%s", feat, "yes"));
		featValIdxer.add(String.format("%s=%s", feat, "no"));
	}

	public void addShapeOneFeatures() {
		String feat = "shape1";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.add(String.format("%s=%s", feat, "ignore"));
		featValIdxer.add(String.format("%s=%s", feat, "noshape"));
	}

	public void addShapeThreeFeatures() {
		String feat = "shape3";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.add(String.format("%s=%s", feat, "ignore"));
		featValIdxer.add(String.format("%s=%s", feat, "noshape"));
	}

	public void addShapeTwoFeatures() {
		String feat = "shape2";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.add(String.format("%s=%s", feat, "ignore"));
		featValIdxer.add(String.format("%s=%s", feat, "noshape"));
	}

	public void addSuffixFeatures(Set<String> suffixes) {
		String feat = "suf";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.add(String.format("%s=%s", feat, "ignore"));
		featValIdxer.add(String.format("%s=%s", feat, "nosuf"));

		for (String suffix : suffixes) {
			featValIdxer.add(String.format("%s=%s", feat, suffix));
		}
	}

	public void addVowelConsonantFeatures() {
		String feat = "vc";
		int idx = featIdxer.getIndex(feat);
		valIdxerMap.put(idx, Generics.newIndexer());

		Indexer<String> featValIdxer = valIdxerMap.get(idx);
		featValIdxer.add(String.format("%s=%s", feat, "ignore"));
		featValIdxer.add(String.format("%s=%s", feat, "others"));
	}

	public void extract(LDocument d) {
		for (LSentence s : d) {
			extract(s);
		}
	}

	public void extract(LDocumentCollection c) {
		for (LDocument d : c) {
			extract(d);
		}
	}

	public void extract(LSentence s) {
		for (int i = 0; i < s.size(); i++) {
			LToken t = s.get(i);
			extractTokenFeatures(t);
		}
		extractChunkFeatures(s);
	}

	private void extractChunkFeatures(LSentence s) {
		if (gzData.size() > 0) {
			List<String> words = s.getTokenStrings(0);
			List<String> feats = Generics.newArrayList(gzData.keySet());
			boolean[][] feat_flags = new boolean[words.size()][feats.size()];

			int win_size = 5;

			int start = 0;
			int end = s.size();

			if (s.get(0).isPadding()) {
				start += 1;
				end -= 1;
			}

			for (int i = start; i < end; i++) {
				for (int j = i + 1; j < Math.min(end, i + win_size); j++) {
					String sub = StrUtils.join(" ", words, i, j);

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

	private void extractTokenFeatures(LToken t) {
		String word = t.getString(0);
		String shape = getShape(word);

		DenseVector F = new DenseVector(0);

		{
			int size = featIdxer.size();
			if (featIdxer.contains("ch")) {
				size += word.length();
			}
			F = new DenseVector(size);
		}

		boolean is_padding = t.isPadding();

		for (int feat_idx = 0; feat_idx < featIdxer.size(); feat_idx++) {
			String feat = featIdxer.getObject(feat_idx);
			Indexer<String> valIdxer = valIdxerMap.get(feat_idx);

			if (feat.equals("word")) {
				int w = -1;
				if (is_padding) {
					w = valIdxer.indexOf(word);
				} else {
					String nw = StrUtils.normalizeNumbers(word.toLowerCase());
					w = is_training ? valIdxer.getIndex(nw) : valIdxer.indexOf(nw, unk);
				}
				F.set(feat_idx, w);
			} else if (feat.equals("pos")) {
				String pos = t.getString(1);
				String val = String.format("%s=%s", feat, pos);
				int val_idx = valIdxer.indexOf(val);

				if (val_idx < 0) {
					val_idx = valIdxer.indexOf(String.format("%s=%s", feat, "nopos"));
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
				String val = "ignore";

				if (!is_padding) {
					Set<Integer> caps = Generics.newHashSet(shape.length());
					for (int i = 0; i < shape.length(); i++) {
						if (shape.charAt(i) == 'A') {
							caps.add(i);
						}
					}

					val = "nocaps";

					if (caps.size() == shape.length()) {
						val = "allcaps";
					} else {
						if (caps.contains(0)) {
							val = "initcap";
						} else {
							val = "hascap";
						}
					}
				}

				F.set(feat_idx, valIdxer.indexOf(String.format("%s=%s", feat, val)));
			} else if (feat.equals("shape1")) {
				if (is_padding) {
					String val = String.format("%s=%s", feat, "ignore");
					int val_idx = valIdxer.indexOf(val);
					F.set(feat_idx, val_idx);
				} else {
					String val = String.format("%s=%s", feat, shape);
					int val_idx = is_training ? valIdxer.getIndex(val) : valIdxer.indexOf(val);
					if (val_idx < 0) {
						val_idx = valIdxer.indexOf(String.format("%s=%s", feat, "noshape"));
					}
					F.set(feat_idx, val_idx);
				}
			} else if (feat.equals("shape2")) {
				if (is_padding) {
					String val = String.format("%s=%s", feat, "ignore");
					int val_idx = valIdxer.indexOf(val);
					F.set(feat_idx, val_idx);
				} else {
					String val = String.format("%s=%s", feat, shape.replaceAll("a+", "a~").replaceAll("A{2,}", "A~A"));
					int val_idx = is_training ? valIdxer.getIndex(val) : valIdxer.indexOf(val);
					if (val_idx < 0) {
						val_idx = valIdxer.indexOf(String.format("%s=%s", feat, "noshape"));
					}
					F.set(feat_idx, val_idx);
				}
			} else if (feat.equals("shape3")) {
				if (is_padding) {
					String val = String.format("%s=%s", feat, "ignore");
					int val_idx = valIdxer.indexOf(val);
					F.set(feat_idx, val_idx);
				} else {
					String suffix = "";
					int suffix_len = 2;

					if (word.length() > 3) {
						int loc = word.length() - suffix_len;
						shape = getShape(word.substring(0, loc));
						suffix = word.substring(loc).toLowerCase();
					}

					String val = String.format("%s=%s", feat, shape.replaceAll("a+", "a~").replaceAll("A{2,}", "A~A"));
					if (suffix.length() > 0) {
						val += "|" + suffix;
					}

					int val_idx = is_training ? valIdxer.getIndex(val) : valIdxer.indexOf(val);

					if (val_idx < 0) {
						val_idx = valIdxer.indexOf(String.format("%s=%s", feat, "noshape"));
					}
					F.set(feat_idx, val_idx);
				}
			} else if (feat.equals("vc")) {
				if (is_padding) {
					String val = String.format("%s=%s", feat, "ignore");
					int val_idx = valIdxer.indexOf(val);
					F.set(feat_idx, val_idx);
				} else {
					StringBuffer sb = new StringBuffer();
					int v_cnt = 0;
					int c_cnt = 0;

					for (int i = 0; i < word.length(); i++) {
						char ch1 = word.charAt(i);
						boolean is_cap = Character.isUpperCase(ch1);
						char ch2 = 'o';

						if (isVowel(ch1)) {
							ch2 = 'v';
							v_cnt++;
						} else if (isConsanant(ch1)) {
							ch2 = 'c';
							c_cnt++;
						} else {
							sb.append("o");
						}

						if (is_cap) {
							ch2 = Character.toUpperCase(ch2);
						}
						sb.append(ch2);
					}

					String val = sb.toString();

					int val_idx = is_training ? valIdxer.getIndex(val) : valIdxer.indexOf(val);

					if (val_idx < 0) {
						val_idx = valIdxer.indexOf(String.format("%s=%s", feat, "others"));
					}
					F.set(feat_idx, val_idx);
				}
			}
		}

		t.setFeatureVector(F);
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
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName, false);
		writeObject(oos);
		oos.flush();
		oos.close();
	}
}
