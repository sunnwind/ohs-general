package ohs.ir.lucene.common;

import java.util.List;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.TermQuery;

import ohs.types.generic.Counter;
import ohs.utils.Generics;

public class AnalyzerUtils {

	public static BooleanQuery getQuery(Counter<String> wordCounts) throws Exception {
		return getQuery(wordCounts, CommonFieldNames.CONTENT);
	}

	public static BooleanQuery getQuery(Counter<String> wordCounts, String field) throws Exception {
		BooleanQuery.Builder builder = new BooleanQuery.Builder();
		List<String> words = wordCounts.getSortedKeys();
		for (int i = 0; i < words.size() && i < BooleanQuery.getMaxClauseCount(); i++) {
			String word = words.get(i);
			double cnt = wordCounts.getCount(word);
			TermQuery tq = new TermQuery(new Term(field, word));
			builder.add(tq, Occur.SHOULD);
		}
		return builder.build();
	}

	public static BooleanQuery getQuery(List<String> words) throws Exception {
		return getQuery(words, CommonFieldNames.CONTENT);
	}

	public static BooleanQuery getQuery(List<String> words, String field) throws Exception {
		BooleanQuery.Builder builder = new BooleanQuery.Builder();

		for (int i = 0; i < words.size(); i++) {
			String word = words.get(i);
			TermQuery tq = new TermQuery(new Term(field, word));
			builder.add(tq, Occur.SHOULD);
		}
		return builder.build();
	}

	public static BooleanQuery getQuery(String text, Analyzer analyzer) throws Exception {
		return getQuery(text, analyzer, CommonFieldNames.CONTENT);
	}

	public static BooleanQuery getQuery(String text, Analyzer analyzer, String field) throws Exception {
		BooleanQuery.Builder builder = new BooleanQuery.Builder();

		Counter<String> c = getWordCounts(text, analyzer);
		List<String> words = c.getSortedKeys();

		for (int i = 0; i < words.size() && i < BooleanQuery.getMaxClauseCount(); i++) {
			String word = words.get(i);
			double cnt = c.getProbability(word);
			TermQuery tq = new TermQuery(new Term(field, word));
			builder.add(tq, Occur.SHOULD);
		}
		return builder.build();
	}

	public static Counter<String> getWordCounts(String text, Analyzer analyzer) throws Exception {
		Counter<String> ret = new Counter<String>();

		TokenStream ts = analyzer.tokenStream(CommonFieldNames.CONTENT, text);
		CharTermAttribute attr = ts.addAttribute(CharTermAttribute.class);
		ts.reset();

		while (ts.incrementToken()) {
			String word = attr.toString();
			ret.incrementCount(word, 1);
		}
		ts.end();
		ts.close();
		return ret;
	}

	public static List<String> getWords(String text, Analyzer analyzer) throws Exception {
		TokenStream ts = analyzer.tokenStream(CommonFieldNames.CONTENT, text);
		CharTermAttribute attr = ts.addAttribute(CharTermAttribute.class);
		ts.reset();

		List<String> ret = Generics.newArrayList();
		while (ts.incrementToken()) {
			String word = attr.toString();
			ret.add(word);
		}
		ts.end();
		ts.close();
		return ret;
	}
}
