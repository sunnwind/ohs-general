package ohs.ml.neuralnet.com;

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

		featIdxer.add("w=<FirstCap>");
		featIdxer.add("w=<AllCaps>");
		featIdxer.add("w=<HashDash>");

		// for (String f : endingNames) {
		// featIdxer.add(f);
		// }
	}

	public IntegerArray extract(MToken t) {
		String word = t.getString(0);

		IntegerArray F = new IntegerArray(new int[featIdxer.size()]);

		if (firstCapPat.matcher(word).matches()) {
			F.set(0, 1);
		}

		if (allCapsPat.matcher(word).matches()) {
			F.set(1, 1);
		}

		if (word.contains("-") || word.contains("_")) {
			F.set(2, 1);
		}

		// for (int i = 0; i < endings.length; i++) {
		// if (endingPatterns[i].matcher(word).matches()) {
		// F.set(i + 3, 1);
		// }
		// }

		word = word.toLowerCase();
		StrUtils.normalizeNumbers(word);

		t.add(word);
		t.add(F);

		return F;

	}

	public Indexer<String> getFeatureIndexer() {
		return featIdxer;
	}
}
