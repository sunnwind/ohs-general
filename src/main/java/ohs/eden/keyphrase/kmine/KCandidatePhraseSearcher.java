 package ohs.eden.keyphrase.kmine;

import java.util.Collection;
import java.util.List;

import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.Token;
import ohs.nlp.ling.types.TokenAttr;
import ohs.tree.trie.hash.Node;
import ohs.tree.trie.hash.Trie;
import ohs.types.common.IntPair;
import ohs.types.generic.ListList;
import ohs.types.generic.Pair;
import ohs.utils.Generics;

public class KCandidatePhraseSearcher {

	public static KCandidatePhraseSearcher newCandidatePhraseSearcher(Collection<String> phrss) {
		Trie<String> dict = new Trie<String>();
		for (String phrs : phrss) {
			MSentence s = MSentence.newSentence(phrs);

			List<String> poss = Generics.newArrayList(s.sizeOfTokens());

			for (Token t : s.getTokens()) {
				poss.add(t.get(TokenAttr.POS));
			}

			Node<String> node = dict.insert(poss);
			node.setFlag(true);
		}
		dict.trimToSize();
		return new KCandidatePhraseSearcher(dict);
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		System.out.printf("ends.");
	}

	private Trie<String> dict;

	public KCandidatePhraseSearcher(Trie<String> dict) {
		this.dict = dict;
	}

	public ListList<IntPair> search(MDocument d) {
		ListList<IntPair> ret = Generics.newListList(d.size());
		for (MSentence s : d) {
			ret.add(search(s));
		}
		return ret;
	}

	public List<IntPair> search(MSentence s) {
		List<Token> ts = s.getTokens();
		List<IntPair> ret = Generics.newArrayList();
		int i = 0;
		while (i < ts.size()) {
			int j = i;

			Node<String> node = dict.getRoot();

			while (j < ts.size()) {
				Token t = ts.get(j);
				String pos = t.get(TokenAttr.POS);

				if (node.hasChild(pos)) {
					node = node.getChild(pos);
					if (node.getFlag()) {
						ret.add(new IntPair(i, j + 1));
					}
				} else {
					break;
				}
				j++;
			}

			if (i == j) {
				i++;
			} else {
				i = j;
			}

		}

		return ret;
	}

	public List<Pair<Integer, Integer>> map(String[] words) {
		List<String> input = Generics.newArrayList(words.length);
		for (String word : words) {
			input.add(word);
		}
		return map(words);
	}

}
