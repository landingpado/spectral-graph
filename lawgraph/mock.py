"""A tiny fake cl.public schema (3 justices, 3 cases, 4 opinions, 1803-1857) for testing without Postgres.
Usage: from lawgraph.mock import make; make(con); SCHEMA.build(con)"""
def make(con):
    con.execute("ATTACH ':memory:' AS cl; CREATE SCHEMA cl.public;")
    mock = {
     'search_court': 'id VARCHAR, short_name VARCHAR, start_date DATE, end_date DATE, date_created TIMESTAMP, date_modified TIMESTAMP',
     'search_docket': 'id BIGINT, court_id VARCHAR, case_name VARCHAR, docket_number VARCHAR, pacer_case_id VARCHAR, date_filed DATE, date_terminated DATE, assigned_to_id BIGINT, referred_to_id BIGINT, date_created TIMESTAMP, date_modified TIMESTAMP',
     'search_opinioncluster': 'id BIGINT, docket_id BIGINT, date_filed DATE, case_name VARCHAR, precedential_status VARCHAR, citation_count INT',
     'search_opinion': 'id BIGINT, cluster_id BIGINT, author_id BIGINT, type VARCHAR, main_version_id BIGINT, plain_text VARCHAR, date_created TIMESTAMP, date_modified TIMESTAMP',
     'search_opinion_joined_by': 'id BIGINT, opinion_id BIGINT, person_id BIGINT',
     'search_opinionscited': 'id BIGINT, citing_opinion_id BIGINT, cited_opinion_id BIGINT, depth INT',
     'people_db_person': 'id BIGINT, name_first VARCHAR, name_middle VARCHAR, name_last VARCHAR, date_dob DATE, date_dod DATE, date_created TIMESTAMP, date_modified TIMESTAMP',
     'people_db_attorney': 'id BIGINT, name VARCHAR, date_created TIMESTAMP, date_modified TIMESTAMP',
     'people_db_party': 'id BIGINT, name VARCHAR, date_created TIMESTAMP, date_modified TIMESTAMP',
     'people_db_role': 'id BIGINT, attorney_id BIGINT, party_id BIGINT, docket_id BIGINT, role INT, role_raw VARCHAR, date_action DATE',
     'people_db_partytype': 'id BIGINT, party_id BIGINT, docket_id BIGINT, name VARCHAR, extra_info VARCHAR, date_terminated DATE',
     'people_db_position': 'id BIGINT, person_id BIGINT, court_id VARCHAR, position_type VARCHAR, job_title VARCHAR, date_start DATE, date_termination DATE, date_retirement DATE, termination_reason VARCHAR, predecessor_id BIGINT, appointer_id BIGINT, supervisor_id BIGINT, date_created TIMESTAMP, date_modified TIMESTAMP',
    }
    for t, cols in mock.items(): con.execute(f'CREATE TABLE cl.public.{t} ({cols})')
    N = "now(), now()"
    con.execute(f"INSERT INTO cl.public.search_court VALUES ('scotus','SCOTUS','1789-09-24',NULL,{N})")
    con.execute(f"INSERT INTO cl.public.people_db_person VALUES (1,'John',NULL,'Marshall','1755-09-24','1835-07-06',{N}),(2,'Joseph',NULL,'Story','1779-09-18','1845-09-10',{N}),(3,'Roger','B.','Taney',NULL,'1864-10-12',{N})")
    con.execute(f"INSERT INTO cl.public.people_db_position VALUES (1,1,'scotus','c-jus',NULL,'1801-02-04','1835-07-06',NULL,NULL,NULL,NULL,NULL,{N}),(2,2,'scotus','jus',NULL,'1812-02-03',NULL,'1845-09-10',NULL,NULL,NULL,NULL,{N}),(3,3,'scotus','c-jus',NULL,'1836-03-28','1864-10-12',NULL,'ded',1,NULL,NULL,{N}),(4,2,'scotus','clerk',NULL,'1800-01-01',NULL,NULL,NULL,NULL,NULL,1,{N}),(5,2,'scotus','ret-jus',NULL,'1845-09-10','1845-09-10',NULL,NULL,NULL,NULL,NULL,{N}),(6,3,'scotus',NULL,'Supreme Court Fellow','1830-01-01','1831-01-01',NULL,NULL,NULL,NULL,NULL,{N}),(7,1,'scotus',NULL,'Interim Judge','1799-01-01','1800-01-01',NULL,NULL,NULL,NULL,NULL,{N})")
    con.execute(f"INSERT INTO cl.public.search_docket VALUES (10,'scotus','Marbury','1',NULL,'1803-02-01','1803-02-24',1,NULL,{N}),(20,'scotus','McCulloch','2','x','1819-03-01','1819-03-06',1,NULL,{N}),(30,'scotus','Dred Scott','3',NULL,'1857-03-01','1857-03-06',3,NULL,{N})")
    con.execute("INSERT INTO cl.public.search_opinioncluster VALUES (100,10,'1803-02-24','Marbury','Published',2),(200,20,'1819-03-06','McCulloch','Published',1),(300,30,'1857-03-06','Dred Scott','Published',0)")
    con.execute(f"INSERT INTO cl.public.search_opinion VALUES (1000,100,1,'020lead',NULL,'...',{N}),(2000,200,1,'020lead',NULL,'...',{N}),(3000,300,3,'020lead',NULL,'...',{N}),(3001,300,2,'040dissent',NULL,'...',{N})")
    con.execute("INSERT INTO cl.public.search_opinionscited VALUES (1,2000,1000,3),(2,3000,1000,1),(3,3000,2000,2),(4,3001,2000,1)")
    con.execute(f"INSERT INTO cl.public.people_db_attorney VALUES (7,'Webster',{N})")
    con.execute(f"INSERT INTO cl.public.people_db_party VALUES (8,'McCulloch',{N}),(9,'Maryland',{N})")
    con.execute("INSERT INTO cl.public.people_db_role VALUES (1,7,8,20,2,'lead',NULL)")
    con.execute("INSERT INTO cl.public.people_db_partytype VALUES (1,8,20,'Plaintiff','',NULL),(2,9,20,'Defendant','',NULL)")
