package com.assignment.submission.AssignmentSubmission.domain;

import javax.persistence.*;

@Entity
public class Assignment
{
	@Id @GeneratedValue(strategy = GenerationType.IDENTITY)
	private long id;
	
	private String status;
	private String githubUrl;
	private String branch;
	private String codeReviewUrl;
	
	@ManyToOne(optional = false)
	private User user;
	
	public User getUser()
	{
		return user;
	}

	public void setUser(User user)
	{
		this.user = user;
	}

	// TODO: private User assignedTo;
	
	public long getId()
	{
		return id;
	}
	
	public void setId(long id)
	{
		this.id = id;
	}
	
	public String getStatus()
	{
		return status;
	}
	
	public void setStatus(String status)
	{
		this.status = status;
	}
	
	public String getGithubUrl()
	{
		return githubUrl;
	}
	
	public void setGithubUrl(String githubUrl)
	{
		this.githubUrl = githubUrl;
	}
	
	public String getBranch()
	{
		return branch;
	}
	
	public void setBranch(String branch)
	{
		this.branch = branch;
	}
	
	public String getCodeReviewUrl()
	{
		return codeReviewUrl;
	}
	
	public void setCodeReviewUrl(String codeReviewUrl)
	{
		this.codeReviewUrl = codeReviewUrl;
	}
	
	
}
