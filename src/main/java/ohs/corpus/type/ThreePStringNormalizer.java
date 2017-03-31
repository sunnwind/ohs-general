package ohs.corpus.type;

import ohs.utils.StrUtils;

public class ThreePStringNormalizer implements StringNormalizer {

	@Override
	public String normalize(String s) {
		StringBuffer sb = new StringBuffer();

		for (String part : s.split(" \\+ ")) {
			String[] tok = part.split(" / ");

			String word = tok[0];
			String pos = tok[1];

			word = StrUtils.normalizeNumbers(word);
			word = StrUtils.normalizeSpaces(word.toLowerCase());

			sb.append(String.format("%s/%s", word, pos));
			sb.append(" ");
		}

		return sb.toString().trim();
	}

}
