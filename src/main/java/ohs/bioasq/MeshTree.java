package ohs.bioasq;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.math.ArrayUtils;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.BidMap;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics;

public class MeshTree {

	public static MeshTree create(String parentChildMapFile, String nameIdMapFile) throws Exception {
		Indexer<String> codeIndexer = Generics.newIndexer();

		CounterMap<Integer, Integer> cm1 = Generics.newCounterMap();
		CounterMap<Integer, Integer> cm2 = Generics.newCounterMap();

		for (String line : FileUtils.readLinesFromText(parentChildMapFile)) {
			String[] parts = line.split(" ");
			String parent = parts[0];
			String child = parts[1];

			int pid = codeIndexer.getIndex(parent);
			int cid = codeIndexer.getIndex(child);

			cm1.setCount(pid, cid, 1);
			cm2.setCount(cid, pid, 1);
		}

		BidMap<Integer, String> idxToName = Generics.newBidMap();

		Set<Integer> extIdxs = Generics.newHashSet();

		for (String line : FileUtils.readLinesFromText(nameIdMapFile)) {
			String[] parts = line.split("=");
			String name = parts[0];
			String code = parts[1];
			int idx = codeIndexer.indexOf(code);

			if (idx == -1) {
				System.out.println(line);
				idx = codeIndexer.getIndex(code);
				extIdxs.add(idx);
			}

			idxToName.put(idx, name);
		}

		for (int node : extIdxs) {
			cm1.setCount(0, node, 1);
			cm2.setCount(node, 0, 1);
		}

		SparseMatrix parentToChild = new SparseMatrix(cm1);
		SparseMatrix childToParent = new SparseMatrix(cm2);

		IntegerArray leaves = new IntegerArray();

		for (int i = 1; i < codeIndexer.size(); i++) {
			if (parentToChild.row(i).size() == 0 && childToParent.row(i).size() >= 0) {
				leaves.add(i);
			}
		}

		return new MeshTree(codeIndexer, idxToName, parentToChild, childToParent, leaves);

	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		MeshTree mt = null;

		boolean create = false;

		if (!create) {
			mt = new MeshTree(MIRPath.BIOASQ_MESH_TREE_SER_FILE);
		} else {
			mt = create(MIRPath.BIOASQ_DIR + "MeSH_parent_child_mapping_2017.txt", MIRPath.BIOASQ_DIR + "MeSH_name_id_mapping_2017.txt");
			mt.writeObject(MIRPath.BIOASQ_MESH_TREE_SER_FILE);
		}

		System.out.println(mt.info());
		System.out.println(mt.getNodesAtLevels());

		System.out.println("process ends.");
	}

	private Indexer<String> codeIndexer;

	private BidMap<Integer, String> idxToName;

	private SparseMatrix parentToChild;

	private SparseMatrix childToParent;

	private IntegerArray leaves;

	private int depth = 1;

	private int cycled_leaf_cnt = 0;

	public MeshTree(Indexer<String> codeIndexer, BidMap<Integer, String> idxToName, SparseMatrix parentToChild, SparseMatrix childToParent,
			IntegerArray leaves) {
		super();
		this.codeIndexer = codeIndexer;
		this.idxToName = idxToName;
		this.parentToChild = parentToChild;
		this.childToParent = childToParent;
		this.leaves = leaves;

		postprocess();
	}

	public MeshTree(ObjectInputStream ois) throws Exception {
		readObject(ois);
	}

	public MeshTree(String fileName) throws Exception {
		readObject(fileName);
	}

	public int depth() {
		return depth;
	}

	public IntegerArray getChildren(int parent) {
		SparseVector row = parentToChild.row(parent);
		IntegerArray ret = new IntegerArray();
		if (row.size() > 0) {
			ret = new IntegerArray(row.indexes());
		}
		return ret;
	}

	public SparseMatrix getChildToParent() {
		return childToParent;
	}

	public String getCode(int idx) {
		return codeIndexer.getObject(idx);
	}

	public Indexer<String> getCodeIndexer() {
		return codeIndexer;
	}

	public BidMap<Integer, String> getIndexToName() {
		return idxToName;
	}

	public String getName(int idx) {
		return idxToName.getValue(idx);
	}

	public IntegerArrayMatrix getNodesAtLevels() {
		CounterMap<Integer, Integer> cm = Generics.newCounterMap();

		for (int leaf : leaves) {
			for (IntegerArray path : getPaths(leaf)) {
				for (int i = 0; i < path.size(); i++) {
					int level = i + 1;
					cm.setCount(level, path.get(i), 1);
				}
			}
		}

		IntegerArrayMatrix ret = new IntegerArrayMatrix();

		for (int i = 0; i < depth; i++) {
			int level = i + 1;
			IntegerArray nodes = new IntegerArray(cm.getCounter(level).keySet());
			nodes.sort(false);
			ret.add(nodes);
		}

		return ret;
	}

	public IntegerArray getParents(int child) {
		SparseVector row = childToParent.row(child);
		IntegerArray ret = new IntegerArray(0);
		if (row.size() > 0) {
			ret = new IntegerArray(row.indexes());
		}
		return ret;
	}

	public SparseMatrix getParentToChild() {
		return parentToChild;
	}

	public IntegerArrayMatrix getPaths(int child) {
		IntegerArrayMatrix paths = new IntegerArrayMatrix();

		IntegerArray prev = new IntegerArray();
		prev.add(child);

		for (int parent : getParents(child)) {
			IntegerArray path = new IntegerArray();
			path.addAll(prev);
			path.add(parent);

			getPaths(path, paths);
		}

		for (IntegerArray path : paths) {
			path.trimToSize();
			ArrayUtils.reverse(path.values());

			// StringBuffer sb = new StringBuffer();
			// for (int i = 0; i < path.size(); i++) {
			// sb.append(String.format("-> %s (%s)", getCode(path.get(i)), getName(path.get(i))));
			// }
			// System.out.println(sb.toString());
		}

		return paths;
	}

	private void getPaths(IntegerArray prev, IntegerArrayMatrix paths) {
		int child = prev.get(prev.size() - 1);

		if (child == 0) {
			paths.add(prev);
		}

		// IntegerArray parents = getParents(child);

		for (int parent : getParents(child)) {
			IntegerArray path = new IntegerArray();
			path.addAll(prev);
			path.add(parent);
			getPaths(path, paths);
		}

		// if (child != 0) {
		// StringBuffer sb = new StringBuffer();
		// for (int i = 0; i < prev.size(); i++) {
		// sb.append(String.format("-> %s (%s)", getCode(prev.get(i)), getName(prev.get(i))));
		// }
		// System.out.println(sb.toString());
		// }
	}

	public int indexOf(String code) {
		return codeIndexer.indexOf(code);
	}

	public int indexOfName(String name) {
		int ret = -1;
		Integer idx = idxToName.getKey(name);
		if (idx != null) {
			ret = idx;
		}
		return ret;
	}

	public String info() {
		StringBuffer sb = new StringBuffer();

		sb.append(String.format("nodes:\t%s\n", codeIndexer.size()));
		sb.append(String.format("leaves:\t%s\n", leaves.size()));
		sb.append(String.format("leaves cycled:\t%s\n", cycled_leaf_cnt));
		sb.append(String.format("depth:\t%s\n", depth));

		return sb.toString();
	}

	private void postprocess() {
		for (int leaf : leaves) {
			IntegerArrayMatrix paths = getPaths(leaf);

			for (IntegerArray path : paths) {
				depth = Math.max(depth, path.size());
			}

			if (paths.size() > 1) {
				cycled_leaf_cnt++;
			}
		}
	}

	public void readObject(ObjectInputStream ois) throws Exception {
		codeIndexer = FileUtils.readStringIndexer(ois);
		parentToChild = new SparseMatrix(ois);
		childToParent = new SparseMatrix(ois);
		leaves = new IntegerArray(ois);

		depth = ois.readInt();
		cycled_leaf_cnt = ois.readInt();

		int size = ois.readInt();
		idxToName = Generics.newBidMap(size);

		for (int i = 0; i < size; i++) {
			idxToName.put(ois.readInt(), ois.readUTF());
		}
	}

	public void readObject(String fileName) throws Exception {
		ObjectInputStream ois = FileUtils.openObjectInputStream(fileName);
		readObject(ois);
		ois.close();
	}

	public int size() {
		return codeIndexer.size();
	}

	public void writeObject(ObjectOutputStream oos) throws Exception {
		FileUtils.writeStringIndexer(oos, codeIndexer);
		parentToChild.writeObject(oos);
		childToParent.writeObject(oos);
		leaves.writeObject(oos);

		oos.writeInt(depth);
		oos.writeInt(cycled_leaf_cnt);

		oos.writeInt(idxToName.size());

		for (int idx : idxToName.getKeys()) {
			String name = idxToName.getValue(idx);
			oos.writeInt(idx);
			oos.writeUTF(name);
		}
	}

	public void writeObject(String fileName) throws Exception {
		ObjectOutputStream oos = FileUtils.openObjectOutputStream(fileName);
		writeObject(oos);
		oos.close();
	}

}
