package ohs.ir.medical.query;

import java.util.List;

import org.apache.lucene.search.Query;

import ohs.ir.medical.general.MIRPath;
import ohs.matrix.SparseVector;

public class TrecCdsQuery implements BaseQuery {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		List<BaseQuery> queries = QueryReader.readTrecCdsQueries(MIRPath.TREC_CDS_2015_QUERY_B_FILE);

		for (int i = 0; i < queries.size(); i++) {
			System.out.println(queries.get(i));
			System.out.println();
		}

		System.out.println("process ends.");
	}

	private String description;

	private String diagnosis;

	private String id;

	private Query luceneQuery;

	private String note;

	private SparseVector queryModel;

	private String summary;

	private String type;

	private List<Integer> words;

	public TrecCdsQuery(String id, String type, String note, String description, String summary, String diagnosis) {
		super();
		this.id = id;
		this.type = type;
		this.note = note;
		this.description = description;
		this.summary = summary;
		this.diagnosis = diagnosis;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		TrecCdsQuery other = (TrecCdsQuery) obj;
		if (description == null) {
			if (other.description != null)
				return false;
		} else if (!description.equals(other.description))
			return false;
		if (id == null) {
			if (other.id != null)
				return false;
		} else if (!id.equals(other.id))
			return false;
		if (summary == null) {
			if (other.summary != null)
				return false;
		} else if (!summary.equals(other.summary))
			return false;
		if (type == null) {
			if (other.type != null)
				return false;
		} else if (!type.equals(other.type))
			return false;
		return true;
	}

	public String getDescription() {
		return description;
	}

	public String getDiagnosis() {
		return diagnosis;
	}

	@Override
	public String getId() {
		return id;
	}

	@Override
	public Query getLuceneQuery() {
		return luceneQuery;
	}

	public String getNote() {
		return note;
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
		String ret = use_desc ? description : summary;
		if (diagnosis != null) {
			ret = ret + "\n" + diagnosis;
		}
		ret = ret.replaceAll("[\\p{Punct}]+", " ");
		return ret.trim();
	}

	public String getSummary() {
		return summary;
	}

	public String getType() {
		return type;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((description == null) ? 0 : description.hashCode());
		result = prime * result + ((id == null) ? 0 : id.hashCode());
		result = prime * result + ((summary == null) ? 0 : summary.hashCode());
		result = prime * result + ((type == null) ? 0 : type.hashCode());
		return result;
	}

	public void setDescription(String description) {
		this.description = description;
	}

	public void setId(String id) {
		this.id = id;
	}

	@Override
	public void setLuceneQuery(Query luceneQuery) {
		this.luceneQuery = luceneQuery;
	}

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append(String.format("qid:\t%s\n", id));
		sb.append(String.format("type:\t%s\n", type));
		sb.append(String.format("note:\t%s\n", note));
		sb.append(String.format("description:\t%s\n", description));
		sb.append(String.format("summary:\t%s\n", summary));
		sb.append(String.format("diagnosis:\t%s", diagnosis));
		return sb.toString();
	}
}
