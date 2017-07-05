package ohs.ir.medical.query;

import java.util.List;

import org.apache.lucene.search.Query;

import ohs.ir.medical.general.MIRPath;
import ohs.matrix.SparseVector;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class TrecPmQuery implements BaseQuery {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		List<BaseQuery> queries = QueryReader.readTrecPmQueries(MIRPath.TREC_PM_2017_QUERY_FILE);

		for (int i = 0; i < queries.size(); i++) {
			System.out.println(queries.get(i));
			System.out.println();
		}

		System.out.println("process ends.");
	}

	private String id;

	private String disease;

	private String gene;

	private String demographic;

	private String other;

	private Query luceneQuery;

	private SparseVector queryModel;

	private List<Integer> words;

	public TrecPmQuery(String id, String disease, String gene, String demographic, String other) {
		super();
		this.id = id;
		this.disease = disease;
		this.gene = gene;
		this.demographic = demographic;
		this.other = other;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		TrecPmQuery other = (TrecPmQuery) obj;
		if (demographic == null) {
			if (other.demographic != null)
				return false;
		} else if (!demographic.equals(other.demographic))
			return false;
		if (disease == null) {
			if (other.disease != null)
				return false;
		} else if (!disease.equals(other.disease))
			return false;
		if (gene == null) {
			if (other.gene != null)
				return false;
		} else if (!gene.equals(other.gene))
			return false;
		if (id == null) {
			if (other.id != null)
				return false;
		} else if (!id.equals(other.id))
			return false;
		if (this.other == null) {
			if (other.other != null)
				return false;
		} else if (!this.other.equals(other.other))
			return false;
		return true;
	}

	public String getDemographic() {
		return demographic;
	}

	public String getDisease() {
		return disease;
	}

	public String getGene() {
		return gene;
	}

	@Override
	public String getId() {
		return id;
	}

	@Override
	public Query getLuceneQuery() {
		return luceneQuery;
	}

	public String getOther() {
		return other;
	}

	@Override
	public List<Integer> getQueryWords() {
		return words;
	}

	@Override
	public String getSearchText() {
		return getSearchText(false);
	}

	public String getSearchText(boolean use_desc) {
		List<String> l = Generics.newArrayList();
		l.add(disease);
		l.add(gene);
		l.add(demographic);
		l.add(other);

		String ret = StrUtils.join("\n", l);
		// ret = ret.replaceAll("[\\p{Punct}]+", " ");
		return ret.trim();
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((demographic == null) ? 0 : demographic.hashCode());
		result = prime * result + ((disease == null) ? 0 : disease.hashCode());
		result = prime * result + ((gene == null) ? 0 : gene.hashCode());
		result = prime * result + ((id == null) ? 0 : id.hashCode());
		result = prime * result + ((other == null) ? 0 : other.hashCode());
		return result;
	}

	public void setDisease(String disease) {
		this.disease = disease;
	}

	public void setGene(String gene) {
		this.gene = gene;
	}

	public void setId(String id) {
		this.id = id;
	}

	@Override
	public void setLuceneQuery(Query luceneQuery) {
		this.luceneQuery = luceneQuery;
	}

	public void setOther(String other) {
		this.other = other;
	}

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append(String.format("qid:\t%s\n", id));
		sb.append(String.format("disease:\t%s\n", disease));
		sb.append(String.format("gene:\t%s\n", gene));
		sb.append(String.format("demographic:\t%s\n", demographic));
		sb.append(String.format("other:\t%s", other));
		return sb.toString();
	}
}
