package ohs.ir.search.app;

import java.util.List;
import java.util.Map;

import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.tree.trie.hash.Node;
import ohs.tree.trie.hash.Trie;
import ohs.utils.Generics;

public class MeSH {

	public static MeSH build(String treeFileName, String descFileName) throws Exception {

		Trie<String> tree = Trie.newTrie();

		List<String> lines = FileUtils.readLinesFromText(treeFileName);
		for (int i = 1; i < lines.size(); i++) {
			String line = lines.get(i);
			String[] parts = line.split("\t");
			String tn = parts[0];
			String descui = parts[1];
			String term = parts[2];

			String[] path = tn.split("\\.");

			Node<String> node = tree.insert(path);

			if (node.getData() == null) {
				node.setData(Generics.newHashMap());
			}

			Map<String, String> data = (Map<String, String>) node.getData();

			data.put("desc_ui", descui);
			data.put("term", term);
		}

		return null;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		MeSH mesh = build(MIRPath.MESH_COL_RAW_DIR + "2017MeshTree.txt", null);

		System.out.println("process ends.");
	}

	private Trie<String> tree;

	public MeSH() {

	}

}
