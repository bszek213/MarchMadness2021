# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 19:05:29 2022

html parse code
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
def html_to_df_web_scrape(URL):
    # URL EXAMPLE: URL = "https://www.sports-reference.com/cbb/schools/clemson/2022-gamelogs.html"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    table = soup.find(id="all_sgl-basic")
    # thead = table.find('thead')
    # th_head = thead.find_all('th')
    # Get body
    tbody = table.find('tbody')
    tr_body = tbody.find_all('tr')
    
    opp_pf = []
    opp_tov = []
    opp_blk = []
    opp_stl  = []
    opp_ast = []
    opp_trb = []
    opp_orb = []
    opp_ft_pct = []
    opp_fta = []
    opp_ft  = []
    opp_fg3_pct = []
    opp_fg3a = []
    opp_fg_pct = []
    opp_fga  = []
    opp_fg  = []
    pf  = []
    tov= []
    blk  = []
    stl = []
    ast  = []
    trb  = []
    orb  = []
    ft_pct = []
    fta = []
    ft = []
    fg3_pct  = []
    game_result = []
    pts = []
    opp_pts = []
    fg = []
    fga = []
    fg_pct = []
    fg3  = []
    for trb in tr_body:
        for td in trb.find_all('td'):
            if td.get('data-stat') == "opp_tov":
                opp_tov.append(td.get_text())
            if td.get('data-stat') == "opp_blk":
                opp_blk.append(td.get_text())
            if td.get('data-stat') == "opp_stl":
                opp_stl.append(td.get_text())
            if td.get('data-stat') == "opp_ast":
                opp_ast.append(td.get_text())
            if td.get('data-stat') == "opp_pf":
                opp_pf.append(td.get_text())
            if td.get('data-stat') == "opp_trb":
                opp_trb.append(td.get_text())
            if td.get('data-stat') == "opp_orb":
                opp_orb.append(td.get_text())
            if td.get('data-stat') == "opp_ft_pct":
                opp_ft_pct.append(td.get_text())
            if td.get('data-stat') == "opp_fta":
                opp_fta.append(td.get_text())
            if td.get('data-stat') == "opp_ft":
                opp_ft.append(td.get_text())
            if td.get('data-stat') == "opp_fg3_pct":
                opp_fg3_pct.append(td.get_text())
            if td.get('data-stat') == "opp_fg3a":
                opp_fg3a.append(td.get_text())
            if td.get('data-stat') == "opp_fg_pct":
                opp_fg_pct.append(td.get_text())
            if td.get('data-stat') == "opp_fga":
                opp_fga.append(td.get_text())
            if td.get('data-stat') == "opp_fg":
                opp_fg.append(td.get_text())
            if td.get('data-stat') == "pf":
                pf.append(td.get_text())
            if td.get('data-stat') == "tov":
                tov.append(td.get_text())
            if td.get('data-stat') == "blk":
                blk.append(td.get_text())
            if td.get('data-stat') == "stl":
                stl.append(td.get_text())
            if td.get('data-stat') == "ast":
                ast.append(td.get_text())
            if td.get('data-stat') == "trb":
                trb.append(td.get_text())
            if td.get('data-stat') == "orb":
                orb.append(td.get_text())
            if td.get('data-stat') == "ft_pct":
                ft_pct.append(td.get_text())
            if td.get('data-stat') == "fta":
                fta.append(td.get_text())
            if td.get('data-stat') == "ft":
                ft.append(td.get_text())
            if td.get('data-stat') == "fg3_pct":
                fg3_pct.append(td.get_text())
            if td.get('data-stat') == "game_result":
                game_result.append(td.get_text())
            if td.get('data-stat') == "pts":
                pts.append(td.get_text())
            if td.get('data-stat') == "opp_pts":
                opp_pts.append(td.get_text())
            if td.get('data-stat') == "fg":
                fg.append(td.get_text())
            if td.get('data-stat') == "fga":
                fga.append(td.get_text())
            if td.get('data-stat') == "fg_pct":
                fg_pct.append(td.get_text())
            if td.get('data-stat') == "fg3":
                fg3.append(td.get_text())
                
    df = pd.DataFrame(list(zip(fga, opp_tov, opp_blk, opp_stl, opp_ast, opp_pf, opp_trb, 
                               opp_orb, opp_ft_pct, opp_fta, opp_ft, opp_fg3_pct,
                               opp_fg3a, opp_fg_pct, opp_fga, opp_fg, pf, tov, blk,
                               stl, ast, trb, orb, ft_pct, fta, ft, fg3_pct, game_result, 
                               pts, opp_pts, fg, fg_pct, fg3)),
                   columns =['fga','opp_tov', 'opp_blk', 'opp_stl', 'opp_ast', 'opp_pf', 'opp_trb', 
                                              'opp_orb', 'opp_ft_pct', 'opp_fta', 'opp_ft', 'opp_fg3_pct',
                                              'opp_fg3a', 'opp_fg_pct', 'opp_fga', 'opp_fg', 'pf', 'tov', 'blk',
                                              'stl', 'ast', 'trb', 'orb', 'ft_pct', 'fta', 'ft', 'fg3_pct', 'game_result', 
                                              'pts', 'opp_pts', 'fg', 'fg_pct', 'fg3'])
    df_return = df.drop(columns=['trb'])
    return df_return

        # # Get data-stat value
        # print(td.get('data-stat'))
  
        # # Get case value
        # print(td.get_text())