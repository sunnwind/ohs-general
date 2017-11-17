package ohs.tree.trie.hash;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import ohs.io.FileUtils;
import ohs.tree.trie.hash.HMTNode.Type;
import ohs.tree.trie.hash.HMTrie.TSResult.MatchType;

public class HMTrie<K> implements Serializable {

	public static class TSResult<K> {

		public enum MatchType {
			EXACT_KEYS_WITH_DATA, PARTIAL_KEYS_WITHOUT_DATA, EXACT_KEYS_WITHOUT_DATA, PARTIAL_KEYS_WITH_DATA, FAIL
		}

		private HMTNode<K> node;

		private MatchType type;

		private int loc;

		public TSResult(HMTNode<K> node, MatchType type, int loc) {
			this.node = node;
			this.type = type;
			this.loc = loc;
		}

		public int getMatchLoc() {
			return loc;
		}

		public HMTNode<K> getMatchNode() {
			return node;
		}

		public MatchType getMatchType() {
			return type;
		}

	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 8031071859567911644L;

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		System.out.println("Process begins.");

		System.out.println("Process ends.");
	}

	public static <K> HMTrie<K> newTrie() {
		return new HMTrie<K>();
	}

	private int depth = 1;

	private int size = 1;

	private HMTNode<K> root = new HMTNode<K>(null, null, null, 0);

	public HMTrie() {

	}

	public void delete(K[] keys) {
		TSResult<K> sr = search(keys);
		if (sr.getMatchType() != MatchType.FAIL) {
			HMTNode<K> node = sr.getMatchNode();
			HMTNode<K> parent = node.getParent();
			parent.getChildren().remove(parent.getKey());
		}
	}

	public TSResult<K> find(K[] keys) {
		return find(keys, 0, keys.length);
	}

	public TSResult<K> find(K[] keys, int start) {
		return find(keys, start, keys.length);
	}

	public TSResult<K> find(K[] keys, int start, int end) {
		return find(Arrays.asList(keys), start, end);
	}

	public TSResult<K> find(List<K> keys) {
		return find(keys, 0, keys.size());
	}

	public TSResult<K> find(List<K> keys, int start) {
		return find(keys, start, keys.size());
	}

	public TSResult<K> find(List<K> keys, int start, int end) {
		HMTNode<K> node = root;
		int num_matches = 0;
		int max_matches = end - start;
		int loc = -1;
		HMTNode<K> tmp = null;

		for (int i = start; i < end; i++) {
			K key = keys.get(i);
			if (node.hasChild(key)) {
				node = node.getChild(key);
				if (node.hasData()) {
					tmp = node;
				}
				loc = i;
				num_matches++;
			} else {
				break;
			}
		}

		MatchType type = MatchType.FAIL;

		if (num_matches == max_matches) {
			if (tmp != null) {
				type = MatchType.EXACT_KEYS_WITH_DATA;
			}
		} else if (num_matches > 0 && num_matches < max_matches) {
			if (tmp != null) {
				type = MatchType.PARTIAL_KEYS_WITH_DATA;
			}
		}

		return new TSResult<K>(tmp, type, loc);
	}

	public HMTNode<K> findLCA(HMTNode<K> node1, HMTNode<K> node2) {
		return null;
	}

	public int getDepth() {
		return depth;
	}

	public List<HMTNode<K>> getLeafNodes() {
		return root.getLeafNodesUnder();
	}

	public Map<Integer, HMTNode<K>> getNodeMap() {
		Map<Integer, HMTNode<K>> ret = new HashMap<Integer, HMTNode<K>>();
		for (HMTNode<K> node : getNodes()) {
			ret.put(node.getID(), node);
		}
		return ret;
	}

	public List<HMTNode<K>> getNodes() {
		return root.getNodesUnder();
	}

	public List<HMTNode<K>> getNodesAtLevel(int level) {
		List<HMTNode<K>> ret = new ArrayList<HMTNode<K>>();
		for (HMTNode<K> node : getNodes()) {
			if (node.getDepth() == level) {
				ret.add(node);
			}
		}
		return ret;
	}

	public HMTNode<K> getRoot() {
		return root;
	}

	public boolean hasKeys(List<K> keys, int start, int end) {
		return null == search(keys, start, end);
	}

	public HMTNode<K> insert(K[] keys) {
		return insert(keys, 0, keys.length);
	}

	public HMTNode<K> insert(K[] keys, int start, int end) {
		return insert(Arrays.asList(keys), start, end);
	}

	public HMTNode<K> insert(List<K> keys) {
		return insert(keys, 0, keys.size());
	}

	public HMTNode<K> insert(List<K> keys, int start, int end) {
		HMTNode<K> node = root;

		for (int i = start; i < end; i++) {
			K key = keys.get(i);
			HMTNode<K> child;
			if (node.hasChild(key)) {
				child = node.getChild(key);
			} else {
				child = new HMTNode<K>(node, key, null, size++);
				node.addChild(child);
			}
			node = child;
		}
		node.incrementCount();

		int d = end - start + 1;
		depth = Math.max(depth, d);
		return node;
	}

	public Set<K> keySet() {
		Set<K> ret = new TreeSet<K>();
		for (HMTNode<K> node : root.getNodesUnder()) {
			if (node.getType() != Type.ROOT) {
				ret.add(node.getKey());
			}
		}
		return ret;
	}

	public Set<K> keySetAtLevel(int level) {
		Set<K> ret = new HashSet<K>();
		for (HMTNode<K> node : getNodes()) {
			if (node.getDepth() == level) {
				ret.add(node.getKey());
			}
		}
		return ret;
	}

	public void read(ObjectInputStream ois) throws Exception {
		depth = ois.readInt();
		size = ois.readInt();

		read(ois, root);
	}

	private void read(ObjectInputStream ois, HMTNode<K> node) throws Exception {
		node.read(ois);
		int size = ois.readInt();

		for (int i = 0; i < size; i++) {
			HMTNode<K> child = new HMTNode<K>();
			child.setParent(node);

			read(ois, child);
			node.addChild(child);
		}
	}

	public void read(String fileName) throws Exception {
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		read(ois);
		ois.close();
	}

	public TSResult<K> search(K[] keys) {
		return search(keys, 0, keys.length);
	}

	public TSResult<K> search(K[] keys, int start) {
		return search(keys, start, keys.length);
	}

	public TSResult<K> search(K[] keys, int start, int end) {
		return search(Arrays.asList(keys), start, end);
	}

	public TSResult<K> search(List<K> keys, int start) {
		return search(keys, start, keys.size());
	}

	public TSResult<K> search(List<K> keys, int start, int end) {
		HMTNode<K> node = root;
		int match_cnt = 0;
		int max_match_cnt = end - start;
		int loc = -1;

		for (int i = start; i < end; i++) {
			K key = keys.get(i);
			if (node.hasChild(key)) {
				node = node.getChild(key);
				loc = i;
				match_cnt++;
			} else {
				break;
			}
		}

		MatchType type = MatchType.FAIL;

		if (match_cnt == max_match_cnt) {
			if (node.getFlag()) {
				type = MatchType.EXACT_KEYS_WITH_DATA;
			} else {
				type = MatchType.EXACT_KEYS_WITHOUT_DATA;
			}
		} else if (match_cnt > 0 && match_cnt < max_match_cnt) {
			if (node.getFlag()) {
				type = MatchType.PARTIAL_KEYS_WITH_DATA;
			} else {
				type = MatchType.PARTIAL_KEYS_WITHOUT_DATA;
			}
		}

		return new TSResult<K>(node, type, loc);
	}

	public int size() {
		return size;
	}

	@Override
	public String toString() {
		return toString(2);
	}

	public String toString(int max_depth) {
		NumberFormat nf = NumberFormat.getInstance();
		StringBuffer sb = new StringBuffer();
		sb.append(String.format("depth:\t%s\n", nf.format(depth)));
		sb.append(String.format("node size:\t%s\n", nf.format(size)));
		toString(root, sb, max_depth);
		return sb.toString().trim();
	}

	private void toString(HMTNode<K> node, StringBuffer sb, int max_depth) {
		if (node.hasChildren() && node.getDepth() < max_depth) {
			int cnt = 0;
			for (HMTNode<K> child : node.getChildren().values()) {
				sb.append("\n");
				for (int j = 0; j < child.getDepth(); j++) {
					sb.append("  ");
				}
				sb.append(String.format("(%d, %d) -> %s", child.getDepth(), cnt++, child.getKey()));
				toString(child, sb, max_depth);
			}
		}
	}

	public void trimToSize() {
		trimToSize(root);
	}

	private void trimToSize(HMTNode<K> node) {
		for (HMTNode<K> child : node.getChildren().values()) {
			trimToSize(child);
		}
		node.trimToSize();
	}

	public void write(ObjectOutputStream oos) throws Exception {
		oos.writeInt(depth);
		oos.writeInt(size);
		write(oos, root);
	}

	private void write(ObjectOutputStream oos, HMTNode<K> node) throws Exception {
		if (node != null) {
			node.write(oos);
			oos.writeInt(node.sizeOfChildren());

			for (HMTNode<K> child : node.getChildren().values()) {
				write(oos, child);
			}
		}
	}

	public void write(String fileName) throws Exception {
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		write(oos);
		oos.close();
	}

}
