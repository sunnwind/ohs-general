/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

/** 
   @author Wei Li <a href="mailto:weili@cs.umass.edu">weili@cs.umass.edu</a>
 */

package ohs.ml.neuralnet.com;

import java.util.regex.Pattern;

import org.jsoup.helper.DescendableLinkedList;

import ohs.matrix.DenseVector;
import ohs.nlp.ling.types.MToken;
import ohs.types.generic.Indexer;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class WordFeatureExtractor {

	private static String[] endings = new String[] { "ing", "ed", "ogy", "s", "ly", "ion", "tion", "ity", "ies" };

	private final String[] endingNames = new String[endings.length];
	private final Pattern[] endingPatterns = new Pattern[endings.length];

	private Indexer<String> featIdxer;

	public WordFeatureExtractor() {
		for (int i = 0; i < endings.length; i++) {
			endingPatterns[i] = Pattern.compile(".*" + endings[i] + "$");
			for (int j = 0; j < 3; j++) {
				endingNames[i] = "W=<END" + endings[i] + ">";
			}
		}

		featIdxer = Generics.newIndexer();

		featIdxer.add("FirstCap");
		featIdxer.add("AllCaps");
		featIdxer.add("HashDash");

		for (String s : endings) {
			featIdxer.add(s);
		}
	}

	private Pattern p1 = Pattern.compile("^\\p{Upper}");

	private Pattern p2 = Pattern.compile("^\\p{Upper}+$");

	public void extract(MToken t) {
		String word = t.getString(0);

		DenseVector F = new DenseVector(featIdxer.size());

		if (p2.matcher(word).matches()) {
			F.add(1, 1);
		} else {
			if (p1.matcher(word).matches()) {
				F.add(0, 1);
			}
			if (word.contains("-") || word.contains("_")) {
				F.add(2, 1);
			}
		}

		for (int i = 0; i < endings.length; i++) {
			if (endingPatterns[i].matcher(word).matches()) {
				F.add(i + 3, 1);
			}
		}

		word = word.toLowerCase();
		StrUtils.normalizeNumbers(word);

		t.add(word);
		t.add(F);

	}
}
