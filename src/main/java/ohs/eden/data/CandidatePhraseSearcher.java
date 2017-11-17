package ohs.eden.data;

import java.util.Collection;
import java.util.List;

import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MToken;
import ohs.tree.trie.hash.HMTNode;
import ohs.tree.trie.hash.HMTrie;
import ohs.types.common.IntPair;
import ohs.types.generic.Pair;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class CandidatePhraseSearcher {

	public static CandidatePhraseSearcher newCandidatePhraseSearcher(Collection<String> kwds) {
		HMTrie<String> dict = new HMTrie<String>();
		for (String pat : kwds) {
			List<String> words = StrUtils.split(" ", pat);
			HMTNode<String> node = dict.insert(words);
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

	public List<List<IntPair>> search(MDocument d) {
		List<List<IntPair>> ret = Generics.newArrayList(d.size());
		for (MSentence s : d) {
			ret.add(search(s));
		}
		return ret;
	}

	public List<IntPair> search(MSentence s) {
		List<IntPair> ret = Generics.newArrayList();
		int i = 0;
		while (i < s.size()) {
			int j = i;

			HMTNode<String> node = dict.getRoot();

			while (j < s.size()) {
				MToken t = s.get(j);
				String word = t.getString(0);
				
				if (node.hasChild(word)) {
					node = node.getChild(word);
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
