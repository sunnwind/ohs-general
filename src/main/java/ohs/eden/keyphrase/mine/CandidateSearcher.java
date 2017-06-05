package ohs.eden.keyphrase.mine;

import java.util.List;

import ohs.nlp.ling.types.KDocument;
import ohs.nlp.ling.types.KSentence;
import ohs.nlp.ling.types.Token;
import ohs.nlp.ling.types.TokenAttr;
import ohs.tree.trie.hash.Node;
import ohs.tree.trie.hash.Trie;
import ohs.types.generic.Pair;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class CandidateSearcher {

	Trie<String> patTrie = new Trie<String>();

	public CandidateSearcher(List<String> posPats) {
		for (String posPat : posPats) {
			List<String> words = StrUtils.split(posPat);
			Node<String> node = patTrie.insert(words);
			node.setFlag(true);
		}
		patTrie.trimToSize();
	}

	public List<Pair<Integer, Integer>> search(KDocument doc) {
		List<Pair<Integer, Integer>> ret = Generics.newArrayList();

		List<List<Token>> sents = doc.getTokens();

		for (int i = 0; i < doc.size(); i++) {
			KSentence sent = doc.get(i);
			List<Token> ts = sent.getTokens();

			int j = 0;
			while (j < ts.size()) {
				int k = j;

				Node<String> node = patTrie.getRoot();

				while (k < ts.size()) {
					Token t = ts.get(k);
					String pos = t.get(TokenAttr.POS);

					if (node.hasChild(pos)) {
						node = node.getChild(pos);
						if (node.getFlag()) {
							ret.add(Generics.newPair(j, k + 1));
						}
					} else {
						break;
					}
					k++;
				}

				if (j == k) {
					j++;
				} else {
					j = k;
				}

			}
		}

		return ret;
	}

}
