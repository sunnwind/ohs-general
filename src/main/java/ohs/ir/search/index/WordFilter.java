package ohs.ir.search.index;

import java.util.Set;
import java.util.regex.Pattern;

import ohs.types.generic.Vocab;
import ohs.utils.Generics;

public class WordFilter {

	private Pattern nonWordPat = Pattern.compile("^\\W+$");

	private Pattern puncPat = Pattern.compile("^\\p{Punct}+");

	private Set<Integer> stopIds = Generics.newHashSet();

	private Set<String> stopwords = Generics.newHashSet();

	private Vocab vocab;

	public WordFilter(Vocab vocab) {
		this(vocab, Generics.newHashSet());
	}

	public WordFilter(Vocab vocab, Set<String> stopwords) {
		super();
		this.vocab = vocab;
		this.stopwords = stopwords;
		buildStopIds();

	}

	public void buildStopIds() {
		stopIds = Generics.newHashSet();

		if (stopwords != null) {
			for (String word : stopwords) {
				int w = vocab.indexOf(word);
				if (w != -1) {
					stopIds.add(w);
				}
			}
		}

		for (int w = 0; w < vocab.size(); w++) {
			String word = vocab.getObject(w);
			if (word.contains("<nu") || puncPat.matcher(word).find() || nonWordPat.matcher(word).find()) {
				stopIds.add(w);
			}
		}

		int unk = vocab.indexOf(Vocab.SYM.UNK.getText());
		if (unk != -1) {
			stopIds.add(unk);
		}

	}

	public void buildStopIds2() {
		stopIds = Generics.newHashSet();

		if (stopwords != null) {
			for (String word : stopwords) {
				int w = vocab.indexOf(word);
				if (w != -1) {
					stopIds.add(w);
				}
			}
		}

		// for (int w = 0; w < vocab.size(); w++) {
		// String word = vocab.getObject(w);
		// if (puncPat.matcher(word).find()) {
		// ret.add(w);
		// }
		// }

		int unk = vocab.indexOf(Vocab.SYM.UNK.getText());
		if (unk != -1) {
			stopIds.add(unk);
		}
	}

	public boolean filter(int w) {
		boolean ret = false;
		if (stopIds.contains(w) || w < 0 || w >= vocab.size()) {
			ret = true;
		}
		return ret;
	}

	public boolean filter(String word) {
		return filter(vocab.indexOf(word));
	}

	public Set<Integer> getStopwords() {
		return stopIds;
	}

	public Vocab getVocab() {
		return vocab;
	}

}
