package com.assignment.submission.AssignmentSubmission.config;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.config.annotation.authentication.builders.AuthenticationManagerBuilder;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
import org.springframework.security.core.userdetails.UserDetailsService;

import com.assignment.submission.AssignmentSubmission.util.CustomPasswordEncoder;

@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter
{
	@Autowired
	private UserDetailsService userDetailsService;
	
	@Autowired
	private CustomPasswordEncoder customPasswordEncoder;
	
	protected void configure(AuthenticationManagerBuilder auth) throws Exception
	{
		auth.userDetailsService(userDetailsService).passwordEncoder(customPasswordEncoder.getPasswordEncoder());
		
		super.configure(auth);
	}
	
	protected void configure(HttpSecurity http) throws Exception
	{
		super.configure(http);
	}
	
	
	
}
