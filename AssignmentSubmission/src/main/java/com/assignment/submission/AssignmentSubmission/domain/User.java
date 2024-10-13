package com.assignment.submission.AssignmentSubmission.domain;

import javax.persistence.*;
import java.time.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;
//import java.util.*;

@Entity
@Table(name="users")
public class User implements UserDetails
{
	private static final long serialVersionUID = 1L;

	@Id @GeneratedValue(strategy = GenerationType.IDENTITY)
	private long id;
	
	private LocalDate cohortStartDate;
	private String username;
	private String password;
//	private List<Authority> authorities = new ArrayList<>();
	
	public long getId()
	{
		return id;
	}
	
	public void setId(long id)
	{
		this.id = id;
	}
	
	public LocalDate getCohortStartDate()
	{
		return cohortStartDate;
	}
	
	public void setCohortStartDate(LocalDate cohortStartDate)
	{
		this.cohortStartDate = cohortStartDate;
	}
	
	@Override
	public String getUsername()
	{
		return username;
	}
	
	public void setUsername(String username)
	{
		this.username = username;
	}
	
	@Override
	public String getPassword()
	{
		return password;
	}
	
	public void setPassword(String password)
	{
		this.password = password;
	}

	@Override
	public Collection<? extends GrantedAuthority> getAuthorities()
	{
		List<GrantedAuthority> roles = new ArrayList<>();
		roles.add(new Authority("ROLE_STUDENT"));
		return roles;
	}
	
    @Override
    public boolean isAccountNonExpired()
    {
        return true;
    }

    @Override
    public boolean isAccountNonLocked()
    {
        return true;
    }

    @Override
    public boolean isCredentialsNonExpired()
    {
        return true;
    }

    @Override
    public boolean isEnabled()
    {
        return true;
    }
	
}
