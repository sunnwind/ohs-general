package ohs.ir.lucene.common;

import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.FilteringTokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

/**
 * Removes stop words from a token stream.
 */
public class NumberStopFilter extends FilteringTokenFilter {

	/**
	 * Builds a Set from an array of stop words, appropriate for passing into the StopFilter constructor. This permits this stopWords
	 * construction to be cached once when an Analyzer is constructed.
	 * 
	 * @param stopWords
	 *            A List of Strings or char[] or any other toString()-able list representing the stopwords
	 * @return A Set ({@link CharArraySet}) containing the words
	 * @see #makeStopSet(java.lang.String[], boolean) passing false to ignoreCase
	 */
	public static CharArraySet makeStopSet(List<?> stopWords) {
		return makeStopSet(stopWords, false);
	}
	/**
	 * Creates a stopword set from the given stopword list.
	 * 
	 * @param stopWords
	 *            A List of Strings or char[] or any other toString()-able list representing the stopwords
	 * @param ignoreCase
	 *            if true, all words are lower cased first
	 * @return A Set ({@link CharArraySet}) containing the words
	 */
	public static CharArraySet makeStopSet(List<?> stopWords, boolean ignoreCase) {
		CharArraySet stopSet = new CharArraySet(stopWords.size(), ignoreCase);
		stopSet.addAll(stopWords);
		return stopSet;
	}

	/**
	 * Builds a Set from an array of stop words, appropriate for passing into the StopFilter constructor. This permits this stopWords
	 * construction to be cached once when an Analyzer is constructed.
	 * 
	 * @param stopWords
	 *            An array of stopwords
	 * @see #makeStopSet(java.lang.String[], boolean) passing false to ignoreCase
	 */
	public static CharArraySet makeStopSet(String... stopWords) {
		return makeStopSet(stopWords, false);
	}

	/**
	 * Creates a stopword set from the given stopword array.
	 * 
	 * @param stopWords
	 *            An array of stopwords
	 * @param ignoreCase
	 *            If true, all words are lower cased first.
	 * @return a Set containing the words
	 */
	public static CharArraySet makeStopSet(String[] stopWords, boolean ignoreCase) {
		CharArraySet stopSet = new CharArraySet(stopWords.length, ignoreCase);
		stopSet.addAll(Arrays.asList(stopWords));
		return stopSet;
	}

	private final CharArraySet stopWords;

	private final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);

	private Pattern p = Pattern.compile("^\\d+[\\d,\\.]*$");

	/**
	 * Constructs a filter which removes words from the input TokenStream that are named in the Set.
	 * 
	 * @param in
	 *            Input stream
	 * @param stopWords
	 *            A {@link CharArraySet} representing the stopwords.
	 * @see #makeStopSet(java.lang.String...)
	 */
	public NumberStopFilter(TokenStream in, CharArraySet stopWords) {
		super(in);
		this.stopWords = stopWords;
	}

	/**
	 * Returns the next input Token whose term() is not a stop word.
	 */
	@Override
	protected boolean accept() {
		if (stopWords.contains(termAtt.buffer(), 0, termAtt.length())) {
			return false;
		}

		String term = termAtt.toString();

		if (p.matcher(term).find()) {
			return false;
		}

		return true;
	}

}
