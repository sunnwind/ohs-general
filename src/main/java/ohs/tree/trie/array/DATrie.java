package ohs.tree.trie.array;

import java.util.List;

import ohs.io.FileUtils;
import ohs.types.number.IntegerArray;
import scala.Char;

public class DATrie {

	public static void main(String[] args) throws Exception {

		System.out.println("process begins.");

		List<String> names = FileUtils.readLinesFromText("../../data/dict/pers.txt");
		DATrie trie = new DATrie();
		trie.test1(names);

		System.out.println("process ends.");
	}

	private IntegerArray base = new IntegerArray(new int[100000]);

	private IntegerArray check = new IntegerArray(new int[100000]);

	private IntegerArray next = new IntegerArray(new int[100000]);

	public void test1(List<String> data) {
		int b = 0;
		int s = 0;

		for (int l = 100; l < data.size(); l++) {
			String q = data.get(l);
			for (int u = 0; u < q.length(); u++) {
				char ch = q.charAt(u);
				int c = Character.codePointAt(q.toCharArray(), u);
				check.set(b + c, s);
				base.set(b + c, base.get(base.get(s) + c));
				
				check.set(base.get(base.get(s) + c) + q, element)

			}

		}

	}
}
