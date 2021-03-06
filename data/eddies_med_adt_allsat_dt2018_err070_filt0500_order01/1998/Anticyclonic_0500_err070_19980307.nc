CDF       
      obs    D   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?h?t?j~?   max       ?öE????       ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N b?   max       P??.       ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ???
   max       =??F       ?   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?Q???R   max       @FU\(?     
?   ?   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ?θQ??     max       @vs\(?     
?  +|   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @*         max       @O?           ?  6   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @?x        max       @?@           6?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ?D??   max       >I?^       7?   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A??   max       B0??       8?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?x?   max       B/?       9?   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?G`   max       C?}Q       :?   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?Nc7   max       C?w?       ;?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?       =   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9       >   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1       ?$   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N b?   max       PJX       @4   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ???`A?7L   max       ??}?H?       AD   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ???
   max       =??F       BT   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?W
=p??   max       @FU\(?     
?  Cd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??         max       @vs\(?     
?  N   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @"         max       @M            ?  X?   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @?x        max       @?*            Y,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D?   max         D?       Z<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ???s??g?   max       ??6??C     @  [L   
   ?      L   
      u         &         <   
                                     I   $               6               
      &            +      
         !   *   v      ?   	               	   $      0      !      O   ,   B   'N?R?P{??N??P?7N?gpNd?fP??.N?? N???OBtN?\Of?O??O?a?O?PQO???O?-?O8y?Of7?O??N?-?N?coOx??NM??OM?PPs??O???N?j?N3??NXָN?\5Pc??N???NcG?O?Ne?aOGZRN b?O???NQ6?NL??N?R?P%(N??O?YN?	0N?vN?j?O❄PD??OC?PN??N˔N?d?NK?Oy??O"??N?e?N???N??O?bO5?]O,?2N??O??{OP
O???N?A????
??t???o?#?
?ě??ě???o??o?D??;D??;??
;??
;?`B<o<#?
<49X<49X<T??<u<?o<??
<??
<??
<??
<??
<?1<?1<?1<?1<ě?<ě?<???<?/<??h<??h<?<?<?<???<???<???=+=t?=t?=??=?w=?w='??=,1=,1=0 ?=49X=@?=]/=]/=u=u=}??=}??=?+=?+=?O?=?\)=?hs=?{=?{=?x?=??F???????????????????????????
,.#??????`afhnqtvqnia````````???HauywqjUNH<)%??20<HU[afffa`VUHG@<22????????????????????"#/<nz????????a=/"ECDGMOR[_dgd[OEEEEEE????????????????????MJEBCIN[gituxythg[NM??????	

???????#)1<IU`cb]I<0.#?????
/<RURQH</#?>KO[fmsh\OC8*????????????????????B=<?HTamz???????maHB)5N[kke[NB) #/<HUahia_UH<3*(# ????????? ????????vorz~??????????????v????????

?????????????????????????????????????????????368BFOPSOB8633333333LO[_hq{???~tha[WSQQL???)7Nj{y)????? 	)5BKOQPLB5 leenz?????????znllllzz??????????zzzzzzzz????????????????????????????????????????????????????	?????CCFHMTTaaehia_THCCCCypnsxz{?????}zyyyyyyUOT[`ht{}{zxwtphb][U)+5754)!105BNX[gtuxytg[NB>71????????????????????????????

??????????????????????????????????????????????????????????????????? ?.NQQYSNB,	2//16@BFOUWTSOEDB=62??????? ???????


?????????mlmn{}??????{ytnmmmm?????????????????????????????????????)5BN[cehh[5???????????????????????????6GOVXSOB5???}|}????????????????}???????

??????29<>IU[_UIC<22222222??????????????????????????#)?????^^aaackmoz????zuma^^*(+/2<HPTTQHA<7/****?@BKNSYTNB??????????[gt???????????tjc]\[??? 	
#0<A>0%#
???????????
??????-./<EHIH</----------???????	????
)-3665-)	__bjnz?????????znea_???????? ???????̺????????ĺ???????????????????????????????????<?C?F?<?4?)????????????????????Ѿ??????	??	???????ھ????????????????????"?/?;?C?????/????????????????????????"E?E?E?E?E?E?FE?E?E?E?E?E?E?E?E?E?E?E?E??нݽ???? ???????ݽܽнǽнннннннп???5?I?a?i?y?t?b?U?N???(???????Ŀֿ??y?????????????????????y?o?m?y?y?y?y?y?y???*?-?6?5?4?*?*????????????????????????????????????????????????????F?S?_?l?x?w?l?_?X?S?F???:?8?:?D?F?F?F?F????????????????????????p?l?e?f?u??????m?y???????????????y?m?`?G?;?9?6?5?@?T?m?G?T?`?k?u?`?T?N?G?;?.?"?????"?*?2?G????????%?)?.?-?'???????ϹȹɹϹܹ???????????	???????????????????????????ŘřŠŭŶŤŜœ?{?n?b?Z?[?`?t?x?w?{ņŘ???????????????????????????????????????'?.?'????????߹ܹԹҹڹ??Z?f?s?y??????z?s?c?d?Z?M?I?B?D?A?7?A?Z??(?4?A?M?Z?\?Z?M?A?;?4?(????????/?;???C?@?;?/?"???	???	????"?*?/?/???????????????????????????????|?{??????????????????????s?u????????????????????Z?d?j?f?b?Z?M?A?4?(???????(?A?M?Z???????????????~?m?T?G????"?C?R?T?m???????
???????????????º²???????????n?zÇÐÎÐÇÆ?z?p?o?n?l?l?l?k?n?n?n?n?????????????????????????????????????????????????????޻????????????????????????ûлۻֻлŻû??????????????????????N?s?????????????????????g?N?A????(?N??#?0?3?<?I?Q?L?I?<?0?#????????????????????????????????????????????ٻ!?-?:?A?F?W?]?S?F?:?-?!????????!?Z?f?h?h?g?f?\?Z?N?M?K?M?N?W?Z?Z?Z?Z?Z?Z?(?4???@?>?4?0?)???????????????(?
?
???
?????????	?
?
?
?
?
?
?
?
?
?
???????ʾپ??????׾ʾ????????????????????T?`?g?j?`?U?T?G?A?E?G?N?T?T?T?T?T?T?T?T?"?.?8?7?.?&?"????? ?"?"?"?"?"?"?"?"?<?H?P?U?H?E?<?6?/?-?,?,?/?1?<?<?<?<?<?<??$?=?I???0????ƳƚƕƔƧƸ????????????'?4?@?H?@?:?4?'???
??????????????"?.?;?G?N?T?V?T?I?G?;?.?"??????????????????????????ŹŹŮŹź???????????ƽĽнݽ????ݽнĽ????????????????ĽĽĽ??o?{ǈǔǡǡǭǳǭǭǡǔǈ?{?{?o?j?k?o?o?H?U?a?zÅÓóù??ìÓ?z?g?L?H?<?8?<?C?H???)?-?O?g?q?t?m?[?B?)?????????????ÇÓàìù??????????ùìàÓÇÅÀÃÇÇ???????????????ؼʼ???????k?Y?f?w???????(?5?A?L?N?Z?g?q?g?Z?N?M?A?5?(?%????(?O?[?h?l?tāāāĀ?t?h?[?V?O?F?H?O?O?O?O?(?5?5?5?2?/?(?????"?(?(?(?(?(?(?(?(???????????#?,?)?#?
?????????????????ؽ???????????????????????????????????????ŔŠŭŶŹ????????????ŹŭŠŖŔŐŐŔŔEEEE(E*E4E0E*EEED?D?D?D?D?EEEE??"?%?/?1?/?"???????????????? ?????????????????????????????????????f?s?v???????????????s?e?Z?X?S?T?Z?e?f????????????????????????????????????????¥???!?-?<?C?E?=?!???????ɺƺɺӺ?????~?????????ź??????????????~?r?m?h?j?r?~???ûлܻ߻??޻Իлû???????????????????D{D?D?D?D?D?D?D?D?D?D?D{DwDrDqDuD{D{D{D{ K ) b 5 < S 7 8 Z $ L 6 # q  4 ` T < R t f \ < ? D 8 i B m I X F Y N b X p  J Y W ] C m '  M \ , 6 T f J n \ e D  2 K ; ! 5 I ,   I  ?     S  j    ~  i  ?  G  ?  ?  ?  "  ?  O  ?  ?  ?  ?  o  ?  ?  F  b  ?    W  ?  \  a  ?  Y    ?  M  ?  ?  V  @  o  R  ?  U  6  u  ?  ?  )  V  N  U  ?    ?  ?    ?  *    8  ?  ?  p  8  ?  ?     =?#?
>$ݼD??=?+;?o?D??=?`B<49X;D??=#?
<?o<??
=?+<?C?=?w<???=o=t?=?w=,1<ě?<ě?=P?`<?/=+=??=ix?<?/<ě?<?=?P=??
=?w=?P=L??=\)=?w=\)=?O?=\)=\)=0 ?=??w=H?9=@?=Y?=<j=??=???>#?
=?C?>I?^=aG?=?\)=q??=???=?\)=?\)=Ƨ?=?t?=?l?=??=??`=???>'??>o>7K?> ĜB"e?BR"B??B??Bq:B ?MBj?B?LB?BٙB#??B&NCB?`B0??B ]RA?? B??B
Ba1B?Ba?B??B
eBBpB??BpMB6RBT?B#B"B]B"??B~?A??A?|?B߮BiB?aB?3B?@B +?B!?B??B
;B9?B??BWB(ȮB ?B??BL?B!φBՅB
?BE9B&?BUDB-??A??B?5B?B
\?B$??Bi?B??B.?B??B??Bs?B"~GB@?B??B?PB?B ?CBàB?iBOBÒB#??B& ?B?<B/?B ;?A?$B ?B`B>?BF9BC?B?B8[B?kB?VB??B?B̃B"?B"Q?B"?IB??A?x?A?{?B??BS|B:lB??BP?B =uB!?MB?>B??B?oB??B@?B(?B?eB?BBK?B"??B?rB
ËBF?B';?B?JB.[}A??B?\B<{B
?B$??B??B??B?mB?B??B??@#/@A?iOAW?RA??zC?}QA,>@A??hA):A??=A?^?@??1@???Ak3RAdet?G`A?ݠA?? A?G(?H??A@vA9?kA???A???AF3`A9?Ai??A???A?z?As1@?]?@?? A???A?B??@ud?A??A5'?A??tAN>
Ag ?A_?%A?t?B?@?{A_??A?#?A'?,B??A?AiA??<A?!?@?2?A???A?-TA?A?1?AܸA??C?k?A?6OA?;pAClmA?)fA???@_Q?@R@???C???@#??A?iWAW+IA?|?C?w?A*?A?x?A'A???A??h@??w@?ذAlAeLn?O??A?auA?]CA҇b?Nc7A@?A;?A???A???AF??A9?Ag;?A??_AȥAs?@?h@???A?|A??yB??@t'A>??A57>A??CAN? Ag?wA_&AÃ?B?@?>Aa?cA?l?A'+B?!A?pAՐ?ÀQ@???A???Aۤ?A??"A?KRA!?A?@?C?h?A??A?<AA2HA?~?A??o@cGm@?l@?^?C??      ?      M         u         &         <   
                                     J   %               7      	         
      '            +      
         !   *   w      ?   	               	   $      0      !      P   ,   B   '      1      5         9                  !   !      !   !                           5   !               5                                 -                  %   -      1                           !                                 )         )                     !                                    #                  1                                 +                  %   #                                                      N?R?O?g?N??Pr?N?gpNd?fP?N?? N???N?? N?\OT?hO??O?a?O???O??\O??PN?e?OP??OH\N?-?N?coO::?NM??OM?POЋO+	+N?j?N3??NXָN?\5PJXN??SNcG?O?Ne?aOGZRN b?O>??NQ6?NL??N?R?P??N??O?YN?	0NF?IN?j?O??O?BOA?O???N˔N?d?NK?Oy??O"??N?e?N???N??O?B?O5?]Oo=N??O}e?OP
O???N?A?    E  \  ?  :  J  
?  ?  ?  [  2    ?  ?  ?  ?  b  3  ?  O  <  ?  [  ?  Q  3  l  N  ?  f  ?    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?    ?  ?  g  Z  	?  l  '  ?  ?  l  ?  p  \    ?  
?  ?  r  ~  ?  I  ?  	4  W  ]???
=q????o<T???ě??ě?=C???o?D??<D??;??
;ě?<?/<o<49X<T??<T??<?9X<?o<?1<??
<??
<???<??
<??
=H?9<???<?1<?1<ě?<ě?<??h<?`B<??h<??h<?<?<?=,1<???<???=+=??=t?=??=?w=#?
='??=<j=??T=8Q?=?x?=@?=]/=]/=u=u=}??=}??=?+=?\)=?O?=??P=?hs=Ƨ?=?{=?x?=??F????????????????????????????????????`afhnqtvqnia````````#/HUagoldUH</#20<HU[afffa`VUHG@<22????????????????????-*.<HUnz?????znUH<1-ECDGMOR[_dgd[OEEEEEE????????????????????JGHNU[]gptttsjg[SNJJ??????	

???????#,4<INU^ba[UI<0# 
/<>DGIH></#
>KO[fmsh\OC8*????????????????????DA>>CHTamz????zmaOHD)5BN[bijc[NB1)2/3<HJUXUSH<22222222???????????????????u{????????????????{u????????

?????????????????????????????????????????????368BFOPSOB8633333333LO[_hq{???~tha[WSQQL??)2AMQME5)	?")5BFKMKEB5-)$leenz?????????znllllzz??????????zzzzzzzz?????????????????????????????????????????????????????????EDGHNTacgha^THEEEEEEypnsxz{?????}zyyyyyyUOT[`ht{}{zxwtphb][U)+5754)!105BNX[gtuxytg[NB>71????????????????????????????? ??????????????????????????????????????????????????????????????????,NRVWQB0)

2//16@BFOUWTSOEDB=62??????? ???????


?????????nmmn{}???{zvnnnnnnn?????????????????????????????????)5BNY[\^\UN5)??????????????????????????%-0,#????}|}????????????????}???????

??????29<>IU[_UIC<22222222??????????????????????????#)?????^^aaackmoz????zuma^^*(+/2<HPTTQHA<7/****?@BKNSYTNB??????????e`_agt???????????tle??? 	
#0<A>0%#
?????????????????-./<EHIH</----------????????
?????
)-3665-)	__bjnz?????????znea_???????? ???????̺????????ĺ???????????????????????????????????????	????	???????????????????Ѿ??????	??	???????ھ???????????????????????"?,?/?0?-?)? ?????????????????????E?E?E?E?E?E?FE?E?E?E?E?E?E?E?E?E?E?E?E??нݽ???? ???????ݽܽнǽннннннн????5?Q?X?\?g?e?N?A?(??????????????y?????????????????????y?o?m?y?y?y?y?y?y???*?-?6?5?4?*?*????????????????????????????????????????????????????F?S?_?l?x?w?l?_?X?S?F???:?8?:?D?F?F?F?F???????????????????????????z?r?m?f?v??m?y?????????????y?m?`?T?G?F?D?E?N?T?^?m?G?T?`?k?u?`?T?N?G?;?.?"?????"?*?2?G????????$?(?-?,?'???????ϹɹϹܹ?????????????????????????????????????????{ŇŠũůŠşŚőŇ?{?n?b?\?]?a?n?w?z?{??????????????????????????????????????????%?&????????????ܹչԹܹ???f?s?????}?w?s?f?[?Z?W?M?H?I?G?E?G?Z?f??(?4?A?M?Z?\?Z?M?A?;?4?(????????/?;???C?@?;?/?"???	???	????"?*?/?/????????????????????????????????????????????????????????s?u????????????????????Z?d?j?f?b?Z?M?A?4?(???????(?A?M?Z???????????????y?m?`?G?;?/?)?)?.?5?G?T?????????
??????????????????????????????n?zÇÐÎÐÇÆ?z?p?o?n?l?l?l?k?n?n?n?n?????????????????????????????????????????????????????޻????????????????????????ûлۻֻлŻû??????????????????????N?s?????????????????????g?N?A????(?N??#?0?1?<?G?G?<?0?#??????????????????????????????????????????????ٻ!?-?:?A?F?W?]?S?F?:?-?!????????!?Z?f?h?h?g?f?\?Z?N?M?K?M?N?W?Z?Z?Z?Z?Z?Z?(?4???@?>?4?0?)???????????????(?
?
???
?????????	?
?
?
?
?
?
?
?
?
?
???????ʾо׾۾ݾ۾׾ʾ??????????????????T?`?g?j?`?U?T?G?A?E?G?N?T?T?T?T?T?T?T?T?"?.?8?7?.?&?"????? ?"?"?"?"?"?"?"?"?<?H?P?U?H?E?<?6?/?-?,?,?/?1?<?<?<?<?<?<??0???A?=?0?????ƳƚƗƧƫƽ????????????'?4?@?H?@?:?4?'???
??????????????"?.?;?G?N?T?V?T?I?G?;?.?"??????????????????????????ŹŹŮŹź???????????ƽĽнݽ????ݽнĽ??????????½ĽĽĽĽĽ??o?{ǈǔǡǡǭǳǭǭǡǔǈ?{?{?o?j?k?o?o?U?a?zÁÇÓîùùìÓ?z?k?P?B?=?<?@?H?U???)?B?Q?^?`?X?B?6?)???????????????Óàìù??????????ùìàÓÇÇÂÇÊÓÓ?????ʼּټܼۼּʼ??????????????????????(?5?A?L?N?Z?g?q?g?Z?N?M?A?5?(?%????(?O?[?h?l?tāāāĀ?t?h?[?V?O?F?H?O?O?O?O?(?5?5?5?2?/?(?????"?(?(?(?(?(?(?(?(???????????#?,?)?#?
?????????????????ؽ???????????????????????????????????????ŔŠŭŶŹ????????????ŹŭŠŖŔŐŐŔŔEEEE(E*E4E0E*EEED?D?D?D?D?EEEE??"?%?/?1?/?"????????????????????????????????????????????????????˾f?s?v???????????????s?e?Z?X?S?T?Z?e?f????????????????????????????????????????¥???!?-?5?>???:?4?!???????׺ں???????~?????????ź??????????????~?r?m?h?j?r?~???ûлܻ߻??޻Իлû???????????????????D{D?D?D?D?D?D?D?D?D?D?D{DwDrDqDuD{D{D{D{ K  b $ < S 6 8 Z  L 0 ) q  3 c . 6 N t f O < ? = 1 i B m I P = Y N b X p  J Y W [ C m ' t M ] , 7 : f J n \ e D  2 F ;  5 : ,   I  ?    S  ?    ~  ?  ?  G  ?  ?  ?    ?  3  i  O  ?  ?  ?  ?  ?  ?  b  ?  ?  j  ?  \  a  ?  ?  ?  ?  M  ?  ?  V  ?  o  R  ?  .  6  u  ?  v  )  ?  ?  '      ?  ?    ?  *    8  f  ?  +  8     ?     =  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?      ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  X  +  ?  ?  ?  `  )  ?  >  ?  	<  
2    ?  D  ?    >  D  3  ?  v  ?  
?  	?    ?  ?  \  Y  V  S  P  L  F  @  9  3  *        ?  ?  ?  ?  ?  ?  ?    ]  ?  ?  ?  ?  ?  ?  ?  o  7     ?  i  ?  ?  ?     ?  :  .  "    
  ?  ?  ?  ?  ?  ?  ?  x  d  H  )  ?  ?  m    J  F  A  =  9  4  0  (      	   ?   ?   ?   ?   ?   ?   ?   ~   j  T  ?  	@  	?  
?  
?  
?  
?  
s  
  	?  	?  ?  ?    i  ^    ?  ?  ?  ?  ?  s  [  E  .    ?  ?  ?  ?  ?  p  P    ?  ?    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  p  ]  H  1      ?  ?  ?  ?    .  M  Z  Y  O  ?  ,    ?  ?  ?  r    d  ?  g  2   ?  2  .  (      	  ?  ?  ?  ?  ?  ?  ?  ?  P  B  3  #                   ?  ?  ?  ?  ?  ?  m  H          ?  ?    ?  p  ?  ?  ?  ?  ?  ?  ?  T    ?  ?  #  ?    r  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  s  j  d  \  O  A  .     ?   ?  ?  :  )    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  -  ?  J  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ]  1  ?  ?  ?  ?  h  *  ?  ?  F  ^  a  \  N  F  C  E  ]  [  A  "  ?  ?  ?    H  ?  ?    ?    J  ?  ?    $  1  1  )      ?  ?  ?  j  2  ?  ?  B  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  j  :  ?  ?  R  ?  ?    *  ?  L  M  B  ,    ?  ?  ?  R    ?  ?  ~  <  ?  ;  P  <  4  -  &                ?  ?  ?  ?  ?  ?  ?  ?  {  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  r  ^  J  5    
   ?   ?    4  M  Z  O  9    ?  ?  y  <  ?  ?  {  6  ?  ?  x  >    ?  ?  ?  ?  ?  ?  ?  ?  v  X  8    ?  ?  ?  c  4     ?   ?  Q  I  ?  5  &      ?  ?  ?  ?  ?  {  ^  <    ?  ?  I   ?  V  w  ?  ?  ?    "  1  -  	  ?  ?  q  G  ?  {    Y     ?  F  O  U  O  L  h  a  N  9  %    ?  ?  ?  L  ?  9  _  ~  ?  N  N  N  N  O  P  Q  Q  Q  Q  R  T  V  \  j  x  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  m  d  [  N  <  )       ?   ?  f  Y  L  ?  4  *           ?  ?  ?  ?  ?  ?  ?  ?  e  H  ?  ?  ?  ?  ?  ?  ?  ?  n  S  9    ?  ?  ?  ?  ?  ?  n  J  ?    ?  ?  ?  ?  r  |  ?  ?  ?  ?  ?  j  =  ?  O  ?  ?   ?  ?  ?  ?  ?  |  l  [  F  -    ?  ?  ?  V    ?  o    ?  n  ?  ?  ?  ?  ?    i  T  6    ?  ?  ?  ?  Z  2  
  ?  ?  <  ?  ?  ?  ?  ?  ?  ?  ?  m  E  !    ?  ?  ?  j  ?    ?  ?  ?  ?  ?  ?  ?  k  L  -    ?  ?  ?  ?  ?  e  E     ?   ?   ?  ?  ?  ?  |  n  ^  M  :  (            ?  ?  ?  ?  ?  s  ?  ?  ?  ?  p  Y  A  *    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  W  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  X    ?  X  ?  @  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  y  l  _  R  D  7  ?    y  t  o  h  `  X  P  H  ?  6  ,  #         ?   ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  j  >    ?  ?  W    ?  	        ?  ?  ?  {  N  "    ?  ?  ?  ?  D  ?  T  ?  $  ?  ?  ?  ?  ?  ?    h  O  5    ?  ?  ?  T    ?  ?  t    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  c  N  8      ?  ?  ?  o  g  Y  J  9  %    ?  ?  ?  ?  v  F  	  ?  ?  M    ?  ]    F  M  U  X  R  K  @  0     
  ?  ?  ?  ?  j  >    ?  ?  ?  	?  	4  	  ?  ?  ?  ?  I  	  ?  y  .  ?  ~    ?  ?  A  ?  ?  M  `  k  `  Q  =  $    ?  ?  ?  H  ?  ?    ?  Y    ?  ?  %  ?  ?  ?      %  #  	  ?  ?  N  ?  j  
?  
#  	  ?  ?  D  ?  ?  ?  ?  ?  ?  ?  }  m  Z  P  D  8  '  	  ?  ?  K  ?  }  ?  ?  Y  ?  "  R  v  ?  ?  ?  ?  J  ?  h  ?     
?  	Q  ?  ?  l  `  U  N  H  C  @  <  .      ?  ?  ?  ?  ?  |  a  D  '  ?  ?  ?  y  _  C  %    ?  ?  ?  |  T  /    ?  ?  ?  d    p  ^  M  ;  *    
  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  c  \  J  4  $        K  @  $    ?  ?  ?  o  K    ?    U        ?  ?  ?  ?  ?  ?  {  _  G  ,    ?  ?  ?  x  *   ?  ?  ?  ?  ?  ?  ?  ?  }  g  O  7      ?  ?  ?  ?  ?  ?  ?  
?  
l  
7  
  	?  	?  	Q  	  ?  s    ?  F  ?  B  ?  ?  ?  ?  x  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  i  M  0    ?  ?  ?  ?  y  Y  ?  n  r  k  [  B  !     ?  ?  ?  a  (  ?  ?    ?  ?  ?  ?  ~  [  !    ?  ?  o  @    ?  ?  ?  u  ;    ?  ?  v  1    c  ?  ?  ?  ?  ?  ?  g  ;  ?  ?  d    ?  V    ?  ?  ?  b  I  F  B  =  4  (      ?  ?  ?  ?  ?  d  C    ?  ?  ?  ?  |  ?  ?  ?  ?  ?  ?  u  3  ?  ?  *  
?  
  	\  ~  ?  ?  i  ?  	4  	"  	  ?  ?  ?  s  9  ?  ?  f    ?  p    |  ?  -  r  ?  W    ?  ?  v  A    
?  
?  
3  	?  	q  ?  l  ?  "  P  Y  n  0  ]  C  &    
?  
?  
?  
m  
6  	?  	?  	g  	  ?  =  ?    {  ?  4