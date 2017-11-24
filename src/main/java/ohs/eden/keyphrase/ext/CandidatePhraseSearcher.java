package ohs.eden.keyphrase.ext;

import java.util.Collection;
import java.util.List;

import ohs.nlp.ling.types.LDocument;
import ohs.nlp.ling.types.LSentence;
import ohs.nlp.ling.types.LToken;
import ohs.tree.trie.hash.HMTNode;
import ohs.tree.trie.hash.HMTrie;
import ohs.types.common.IntPair;
import ohs.types.generic.Pair;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class CandidatePhraseSearcher {

	public static CandidatePhraseSearcher newCandidatePhraseSearcher(Collection<String> pats) {
		HMTrie<String> dict = new HMTrie<String>();
		for (String pat : pats) {
			List<String> poss = StrUtils.split(" ", pat);
			HMTNode<String> node = dict.insert(poss);
			node.setFlag(true);
		}
		dict.trimToSize();
		return new CandidatePhraseSearcher(dict);
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		System.out.printf("ends.");
	}

	private HMTrie<String> dict;

	public CandidatePhraseSearcher(HMTrie<String> dict) {
		this.dict = dict;
	}

	public List<List<IntPair>> search(LDocument d) {
		List<List<IntPair>> ret = Generics.newArrayList(d.size());
		for (LSentence s : d) {
			ret.add(search(s));
		}
		return ret;
	}

	public List<IntPair> search(LSentence s) {
		List<IntPair> ret = Generics.newArrayList();
		int i = 0;
		while (i < s.size()) {
			int j = i;

			HMTNode<String> node = dict.getRoot();

			while (j < s.size()) {
				LToken t = s.get(j);
				String pos = t.getString(1);

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
