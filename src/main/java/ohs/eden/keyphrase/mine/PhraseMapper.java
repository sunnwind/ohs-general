package ohs.eden.keyphrase.mine;

import java.util.Collection;
import java.util.List;

import ohs.corpus.type.RawDocumentCollection;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.tree.trie.hash.Node;
import ohs.tree.trie.hash.Trie;
import ohs.types.generic.Counter;
import ohs.types.generic.Pair;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class PhraseMapper {

	public static Trie<String> createDict(Collection<String> phrss) {
		Trie<String> ret = new Trie<String>();
		for (String phrs : phrss) {
			List<String> words = StrUtils.split(phrs);

			if (words.size() > 1) {
				Node<String> node = ret.insert(words);
				node.setFlag(true);
			}
		}
		ret.trimToSize();
		return ret;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		Counter<String> c = FileUtils.readStringCounterFromText("../../data/medical_ir/trec_cds/2014/phrs/kwds.txt.gz");

		Trie<String> dict = PhraseMapper.createDict(c.keySet());
		PhraseMapper m = new PhraseMapper(dict);

		{
			RawDocumentCollection rdc = new RawDocumentCollection(MIRPath.TREC_CDS_2014_COL_DC_DIR);
			System.out.println(rdc.getAttrData());
			System.out.println(rdc.size());

			for (int i = 0; i < 100; i++) {
				List<String> vals = rdc.get(i);

				String body = vals.get(3);

				List<String> words = StrUtils.split(body);

				List<Pair<Integer, Integer>> ps = m.map(words);

				for (Pair<Integer, Integer> p : ps) {
					System.out.printf("%s, [%s]\n", p, StrUtils.join(" ", words, p.getFirst(), p.getSecond()));
				}
				System.out.println();
			}
		}

		System.out.printf("ends.");
	}

	private Trie<String> dict;

	public PhraseMapper(Trie<String> dict) {
		this.dict = dict;

	}

	public List<Pair<Integer, Integer>> map(List<String> words) {
		List<Pair<Integer, Integer>> ret = Generics.newArrayList();
		int i = 0;
		while (i < words.size()) {
			int j = i;

			Node<String> node = dict.getRoot();

			while (j < words.size()) {
				String key = words.get(j);
				if (node.hasChild(key)) {
					node = node.getChild(key);
					if (node.getFlag()) {
						ret.add(Generics.newPair(i, j + 1));
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

}
