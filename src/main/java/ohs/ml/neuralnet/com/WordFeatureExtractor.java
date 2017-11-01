package ohs.ml.neuralnet.com;

import java.util.Set;
import java.util.regex.Pattern;

import ohs.nlp.ling.types.MToken;
import ohs.types.generic.Indexer;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class WordFeatureExtractor {

	private static String[] endings = new String[] { "ing", "ed", "ogy", "s", "ly", "ion", "tion", "ity", "ies" };

	private Pattern allCapsPat = Pattern.compile("^[\\p{Upper}]+$");
	private final String[] endingNames = new String[endings.length];

	private final Pattern[] endingPatterns = new Pattern[endings.length];

	private Indexer<String> featIdxer;

	private Pattern firstCapPat = Pattern.compile("^[\\p{Upper}].*");

	public WordFeatureExtractor() {
		for (int i = 0; i < endings.length; i++) {
			endingPatterns[i] = Pattern.compile(".*" + endings[i] + "$");
			for (int j = 0; j < 3; j++) {
				endingNames[i] = "w=<END" + endings[i] + ">";
			}
		}

		featIdxer = Generics.newIndexer();

		featIdxer.add("w=<Noninfo>");
		featIdxer.add("w=<Lowercase>");
		featIdxer.add("w=<AllCaps>");
		featIdxer.add("w=<FirstCap>");
		featIdxer.add("w=<MixedCaps>");

		// for (String f : endingNames) {
		// featIdxer.add(f);
		// }
	}

	public IntegerArray extract(MToken t) {
		String word = t.getString(0);

		IntegerArray F = new IntegerArray(new int[featIdxer.size()]);

		Set<Integer> caps = Generics.newHashSet();
		Set<Integer> lowers = Generics.newHashSet();

		for (int i = 0; i < word.length(); i++) {
			char ch = word.charAt(i);
			if (Character.isUpperCase(ch)) {
				caps.add(i);
			} else if (Character.isLowerCase(ch)) {
				lowers.add(i);
			}
		}

		// all caps
		if (caps.size() == word.length()) {
			F.set(featIdxer.indexOf("w=<AllCaps>"), 1);
		}

		// first cap
		if (caps.contains(0)) {
			F.set(featIdxer.indexOf("w=<FirstCap>"), 1);
		}

		// mixed caps
		if (!caps.contains(0) && caps.size() > 0) {
			F.set(featIdxer.indexOf("w=<MixedCaps>"), 1);
		}

		// all lowercases
		if (lowers.size() == word.length()) {
			F.set(featIdxer.indexOf("w=<Lowercase>"), 1);
		}

		if (caps.size() == 0 && lowers.size() == 0) {
			F.set(featIdxer.indexOf("w=<Noninfo>"), 1);
		}

		word = word.toLowerCase();
		word = StrUtils.normalizeNumbers(word);

		t.add(word);
		t.add(F);

		return F;

	}

	public Indexer<String> getFeatureIndexer() {
		return featIdxer;
	}
}
