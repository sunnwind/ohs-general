package ohs.ml.neuralnet.com;

import java.util.List;
import java.util.Set;

import ohs.matrix.DenseVector;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MToken;
import ohs.types.generic.Indexer;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class WordFeatureExtractor {

	private Indexer<String> featIdxer;

	private DenseVector featTypeSizes;

	public WordFeatureExtractor() {

	}

	public static final String[] FEAT_TYPES = { "cap", "suf", "per", "org", "loc", "misc" };

	public WordFeatureExtractor(Set<String> caps, Set<String> suffixes, Set<String> pers, Set<String> orgs,
			Set<String> locs, Set<String> miscs) {

		featIdxer = Generics
				.newIndexer(caps.size() + pers.size() + orgs.size() + locs.size() + miscs.size() + suffixes.size());

		featTypeSizes = new DenseVector(FEAT_TYPES.length);

		{

			List<Set<String>> F = Generics.newArrayList();
			F.add(caps);
			F.add(suffixes);
			F.add(pers);
			F.add(orgs);
			F.add(locs);
			F.add(miscs);

			for (int i = 0; i < FEAT_TYPES.length; i++) {
				String type = FEAT_TYPES[i];
				Set<String> feats = F.get(i);

				for (String feat : feats) {
					feat = feat.replaceAll("[\\s\u2029]+", "").trim();
					featIdxer.add(String.format("%s=%s", type, feat));
				}

				featTypeSizes.add(i++, feats.size());
			}
		}
	}

	public void extract(MSentence s) {

		for (int i = 0; i < s.size(); i++) {
			MToken t = s.get(i);
			t.add(t.getString(0).toLowerCase());
			t.add(new DenseVector(FEAT_TYPES.length));
			extractTokenFeatures(t);
		}

		extractChunkFeatures(s);
	}

	public void extractChunkFeatures(MSentence s) {

		List<String> words = s.getTokenStrings(s.get(0).size() - 2);
		boolean[][] feat_flags = new boolean[words.size()][4];

		int win_size = 5;

		for (int i = 0; i < s.size(); i++) {
			for (int j = i + 1; j < Math.min(s.size(), i + win_size); j++) {
				String substr = StrUtils.join("", words, i, j);

				for (int k = 2; k < FEAT_TYPES.length; k++) {
					String feat = String.format("%s=%s", FEAT_TYPES[k], substr);

					if (featIdxer.contains(feat)) {
						for (int u = i; u < j; u++) {
							feat_flags[u][k - 2] = true;
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
				if (flags[j]) {
					F.set(j, 1);
				}
			}

		}
	}

	public void extractTokenFeatures(MToken t) {
		String word = t.getString(0);

		Set<Integer> caps = Generics.newHashSet(word.length());

		for (int i = 0; i < word.length(); i++) {
			char ch = word.charAt(i);
			if (Character.isUpperCase(ch)) {
				caps.add(i);
			}
		}

		DenseVector F = (DenseVector) t.get(t.size() - 1);

		int feat_type_idx = 0;

		{
			String feat = "nocaps";

			if (caps.size() == word.length()) {
				feat = "allcaps";
			} else {
				if (caps.contains(0)) {
					feat = "initcap";
				} else {
					feat = "hascap";
				}
			}

			int f = featIdxer.indexOf(String.format("%s=%s", FEAT_TYPES[feat_type_idx], feat));
			F.add(feat_type_idx, f);
		}

		feat_type_idx++;

		word = word.toLowerCase();

		{
			List<String> suffixes = Generics.newArrayList();
			for (int j = 0; j < word.length(); j++) {
				String suffix = String.format("%s=%s", FEAT_TYPES[1],
						word.substring(word.length() - j - 1, word.length()));

				if (featIdxer.contains(suffix)) {
					suffixes.add(suffix);
				}
			}

			if (suffixes.size() > 0) {
				F.add(feat_type_idx, featIdxer.indexOf(suffixes.get(suffixes.size() - 1)));
			} else {
				F.add(feat_type_idx, -1);
			}
		}
	}

	public Indexer<String> getFeatureIndexer() {
		return featIdxer;
	}
}
