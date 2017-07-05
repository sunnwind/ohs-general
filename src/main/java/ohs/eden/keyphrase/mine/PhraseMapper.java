package ohs.eden.keyphrase.mine;

import java.util.Collection;
import java.util.List;

import ohs.corpus.type.RawDocumentCollection;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.tree.trie.hash.Node;
import ohs.tree.trie.hash.Trie;
import ohs.types.common.IntPair;
import ohs.types.generic.Counter;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class PhraseMapper<K> {

	public static Trie<String> createTrie(Collection<String> phrss) {
		Trie<String> ret = new Trie<String>();
		for (String phrs : phrss) {
			List<String> words = StrUtils.split(phrs);
			Node<String> node = ret.insert(words);
			node.setFlag(true);
		}
		ret.trimToSize();
		return ret;
	}

	public static Trie<Integer> createTrie(Collection<String> phrss, Vocab vocab) {
		Trie<Integer> ret = new Trie<Integer>();
		for (String phrs : phrss) {
			List<String> words = StrUtils.split(phrs);
			List<Integer> ws = vocab.indexesOfKnown(words);

			if (ws.size() > 0 && words.size() == ws.size()) {
				Node<Integer> node = ret.insert(ws);
				node.setFlag(true);
			}
		}
		ret.trimToSize();
		return ret;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		System.out.printf("ends.");
	}

	private Trie<K> dict;

	public PhraseMapper(Trie<K> dict) {
		this.dict = dict;
	}

	public List<Pair<Integer, Integer>> map(K[] words) {
		List<K> input = Generics.newArrayList(words.length);
		for (K word : words) {
			input.add(word);
		}
		return map(words);
	}

	public List<IntPair> map(List<K> words) {
		List<IntPair> ret = Generics.newArrayList();
		int i = 0;
		while (i < words.size()) {
			int j = i;

			Node<K> node = dict.getRoot();

			while (j < words.size()) {
				K key = words.get(j);
				if (node.hasChild(key)) {
					node = node.getChild(key);
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

	public void test1() throws Exception {
		Counter<String> c = FileUtils.readStringCounterFromText("../../data/medical_ir/trec_cds/2014/phrs/kwds.txt.gz");

		Trie<String> dict = PhraseMapper.createTrie(c.keySet());
		PhraseMapper<String> m = new PhraseMapper<String>(dict);

		{
			RawDocumentCollection rdc = new RawDocumentCollection(MIRPath.TREC_CDS_2014_COL_DC_DIR);
			System.out.println(rdc.getAttrData());
			System.out.println(rdc.size());

			for (int i = 0; i < 100; i++) {
				List<String> vals = rdc.get(i);

				String body = vals.get(3);

				List<String> words = StrUtils.split(body);

				List<IntPair> ps = m.map(words);

				for (Pair<Integer, Integer> p : ps) {
					System.out.printf("%s, [%s]\n", p, StrUtils.join(" ", words, p.getFirst(), p.getSecond()));
				}
				System.out.println();
			}
		}
	}

}
