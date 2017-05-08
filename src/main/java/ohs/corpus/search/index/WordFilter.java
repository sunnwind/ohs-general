package ohs.corpus.search.index;

import java.util.Set;
import java.util.regex.Pattern;

import ohs.types.generic.Vocab;
import ohs.utils.Generics;

public class WordFilter {

	private Vocab vocab;

	private Set<Integer> stopwords = Generics.newHashSet();

	private Pattern puncPat = Pattern.compile("^\\p{Punct}+");

	private Pattern nonWordPat = Pattern.compile("^\\W+$");

	public WordFilter(Vocab vocab) {
		this(vocab, Generics.newHashSet());
	}

	public WordFilter(Vocab vocab, Set<String> stopwords) {
		super();
		this.vocab = vocab;
		this.stopwords = buildStopwords(stopwords);

	}

	private Set<Integer> buildStopwords(Set<String> stopwords) {
		Set<Integer> ret = Generics.newHashSet(stopwords.size());
		
		if (stopwords != null) {
			for(String word : stopwords){
				int w = vocab.indexOf(word);
				if(w != -1){
					ret.add(w);
				}
			}
		}

		for (int w = 0; w < vocab.size(); w++) {
			String word = vocab.getObject(w);
			if (word.contains("<nu") || puncPat.matcher(word).find() || nonWordPat.matcher(word).find()) {
				ret.add(w);
			}
		}

		int unk = vocab.indexOf(Vocab.SYM.UNK.getText());
		if (unk != -1) {
			ret.add(unk);
		}
		return ret;
	}

	public boolean filter(int w) {
		boolean ret = false;
		if (stopwords.contains(w) || w < 0 || w >= vocab.size()) {
			ret = true;
		}
		return ret;
	}

	public boolean filter(String word) {
		return filter(vocab.indexOf(word));
	}

	public Set<Integer> getStopwords() {
		return stopwords;
	}

	public Vocab getVocab() {
		return vocab;
	}

}
