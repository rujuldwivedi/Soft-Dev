package com.assignment.submission.AssignmentSubmission.domain;

import org.springframework.security.core.GrantedAuthority;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.ManyToOne;

@Entity
public class Authority implements GrantedAuthority
{
	private static final long serialVersionUID = 1L;
	
	@Id @GeneratedValue(strategy = GenerationType.IDENTITY)
	private long id;
	private String authority;
	
	@ManyToOne(optional = false)
	private User user;
	
	public Authority (String authority)
	{
		this.authority = authority;
	}
	
	public long getId()
	{
		return id;
	}
	
	public void setId(long id)
	{
		this.id = id;
	}
	
	@Override
	public String getAuthority()
	{
		return authority;
	}
	
	public void setAuthority(String authority)
	{
		this.authority = authority;
	}
	
	public User getUser()
	{
		return user;
	}
	
	public void setUser(User user)
	{
		this.user = user;
	}
	
}
