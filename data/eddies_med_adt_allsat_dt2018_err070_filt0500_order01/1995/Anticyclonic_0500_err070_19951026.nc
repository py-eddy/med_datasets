CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?h?t?j~?   max       ??I?^5?}      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N8?   max       P?	?      ?  ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ?+   max       =?S?      ?  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>??\)   max       @F?Q??     ?   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??(?\    max       @v?fffff     ?  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @.         max       @N            p  1?   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @??        max       @?ܠ          ?  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ????   max       >?1'      ?  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A???   max       B/??      ?  3?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A???   max       B/?      ?  4?   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ????   max       C??!      ?  5?   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =	J?   max       C??      ?  6?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?      ?  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      ?  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          '      ?  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N8?   max       P
??      ?  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ???*?0?   max       ??w1???      ?  :?   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ?o   max       >?w      ?  ;?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>??\)   max       @F?Q??     ?  <?   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??(?\    max       @v?=p??
     ?  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @&         max       @M?           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @??        max       @???          ?  N?   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A^   max         A^      ?  O?   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ??-?qv   max       ??jOv`     ?  Pl                                 +      G   )                  :      	               &            %      G   ?   >   "      H         i   &   
            )      -   	   0      *   9   2   $N???N???NJ?8N???OKhO?^?Oi}ZOT?|O???NM?PסN-TP3??O??[O.?dN8?O%7?O2??O?%P	MO??^O??OOZ?N?ZO:n?O??O??N?7?N??AN+?nO???OWM%P??P?	?P&O?VN?i<P??N?GO?uP6O?ǄNs?N/.O?nON?_?O?BbO,\?O+_N??Ob|?N??O?$?O?TnO9?O7???+??/?e`B?D????o:?o;o<t?<49X<u<?t?<?1<?j<ě?<???<???<?/<?`B<??h<??h<??h<???=o=+=+=C?=\)=\)=\)=\)=?w=0 ?=<j=<j=@?=@?=H?9=H?9=P?`=P?`=P?`=P?`=ix?=m?h=m?h=q??=u=u=}??=?o=?o=?+=?t?=?-=?^5=?S??????????????????????????????????????GKOSY[bhkhf[WOGGGGGGqtv?????????utqqqqqq#AHNNUZaca`UH<2##yvw|~?????????????~y???????????????????)57BFGB<5))Bht|}y~th[JC6ST`amnzzzymaaTSSSSSS94631:HTmy{soomaTH9????????????????????????????
	????????
#'/9Unx???jaUH<#
"$/;HQHFCD=;.'")),)*56?CHGCA6*???????????????????????????????????????????????????????????????????? ????????????????????????????'%&),6BOW[_`_^[OF9)'fghty???~tpgffffffff???????????????????????)BK@6)?!"#%/<HUaknngUHD</$!#/49<>=<93/##?????????

?????


#,.#










MLLLNXh??????sih[QOM?????????
???????????)2:@>5-) ??)5Nt???????g[B"?????
)5?BA>5)?????
#0<IUUPI50#
????
"
 ??????????#/EAF;?=@#
??	
#$#$#"
 	#)/2665/#
????????????????????????)389?)????????????????????????}??????????}}}}}}}}????????

????????=<?BHO[\][WOLB======????????????????????????????????????????UOPZ[ht????????{th[U11358?BNQUSONFB51111?????????????????????????????????????????????????????????????????????????????????????
##)+&#
????0572/$#
	 
"#0ŇŔŠŭŶŹźŹŭŠŔŇ?{?x?n?n?q?{ņŇ?O?O?O?N?O?X?O?B?6?*?6?9?B?G?O?O?O?O?O?O?y?????????????????????{?y?s?y?y?y?y?y?yÇÑÓÙÖÓÌÇÂ?z?w?w?z?{ÇÇÇÇÇÇ?????!???	???????????????????????[?h?tčĚĢĥĤĚā?t?[?I?E?B?>?B?E?O?[?????????????????????????????????????????T?`?n?x?w?u?n?m?`?T?G???;?9?6?8?;?@?G?T???????????????????y?`?T?C?@?2?T?m?~?????ѿտݿݿ޿ݿҿѿɿĿ??ĿĿпѿѿѿѿѿ??????
?#?<?I?L?I?G?<?#??
??????ļĸĹ???/?<?C?F?<?/?#?"?#?,?/?/?/?/?/?/?/?/?/?/????(?5?=?B?B?X?M?8???н????????½ݽ????2?A?Q?Z?g?c?Z?N?A?5????? ??	???T?]?a?l?m?q?s?q?m?[?T?H?;?/?,?/?;?=?G?T?f?l?j?g?f?Z?W?U?Z?^?f?f?f?f?f?f?f?f?f?f?`?m?y???????????????y?m?c?`?V?O?P?T?\?`?F?9?<?:?8?-?!??????????????-?:?F?F?\?h?u?w?~Ɓ?~?u?o?h?\?O?C?=?=?@?K?O?V?\?????????3?,?????????ŵŲŴŹŽ??????(?5?A?S?\?a?Z?N?5?(???????????????m?n?y?????????????????y?m?`?]?V?X?`?b?m?ʾ׾????????????????׾ʾ????????????Ⱦʿ????ĿǿĿ??????????????????????????????N?Z?]?c?g?q?s???????s?g?N?=?5?,?.?5?A?N?-?F?x???????????x?m?g?Q?:?-??????-?(?A?M?Z?y???}?z?t?f?Z?M?A?+??????(?=?I?V?b?o?y?z?o?b?]?V?I?C?=?0?'?0?0?=?=???ûлٻܻ??ܻԻлû??????????????????????????????????y????????????????????????f?r???????????????????r?f?Y?V?M?Y?]?f?y???????????????????y?l?g?`?Y?V?W?a?l?y?
?#?<?I?S?X?R?E?0???????Ŀĵ?????????
?hāĠĲıġėč?t?[?6?????????)?B?h¿?????
?????????²?t?d?^?b?t²¿?ʼּ????????????????ʼ??????ļ?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D??B?[¦°°¸²¦?t?g?[?N?-?"? ?'?B?????	???"?"?"???	???????????????????T?`?m?y?y?x?q?m?e?`?T?G?;?1?0?;?@?G?O?T???ɺֺ???????????ɺ????????????????????????????????~?r?e?Y?E?@?A?9?@?Y?~?????
???????
????????????????????????	????	?????????????????????????????????	??? ?%?(?%?"???????????????????ܻ????????	????????߻ܻӻܻܻܻܻܻ?ù????????????????????ùìÓÈÊÓÜêù?H?U?a?zÅ?z?y?n?`?U?H?<?/?,?/?0?6?<?C?H?3?@?L?Y?Z?e?m?m?e?c?Y?L?G?@?6?3?0?.?0?3?0?=?I?O?V?b?h?e?b?V?I?=?<?0?)?$?0?0?0?0?????ʼּؼ߼߼ۼмʼ????????????????????	???????	? ???????????	?	?	?	?	?	?4?@?M?Y?f?r?|?f?_?K?3?'???????'?4F1F=FJFVFcFqFxFrFoFcFVFJF=F$FFFFFF1E?E?E?E?E?E?E?E?E?E?E?E?EuErEnEpEuEvE{E??B?6?*?????
????*?6?C?O?T?[?O?N?B L ^ a G 2 2 . 3 W S V H 8 E : w  ( (  * ) 8 : M x 5 Q S X < O L 4 \ C 3 6 B M 4 2 [ 0 o Z O G ? [ 0 @ \ = 0 %  ?  r  ?  ?  ?  V  ?  ?  ?  ?  ?  2  J  I  |  k  d  ?  ?  X  ?  M  ?  4  ?  h  ?  #  +  h  W  ?  ?  ?    n  ?  ?  ?  W  P  o  ?  M  ?  ?  b  ?  ~  ?  ?  ?  c    ?  ????????
?#?
;o<?`B<???<T??<?/=o<?t?=y?#<?/=??=?o=??<?`B=,1=m?h=,1=? ?=aG?=#?
=8Q?=?P=L??=m?h=??=P?`=T??='??=???=u=??h>?1'=?;d=??T=??=???=m?h=?hs>?w=?9X=?7L=y?#=???=?\)=??=??
=?"?=?t?=?`B=?hs=?l?>t?>\)>?+B?B?DBQB
{8B͠B ??Bd!B?3B?kA??.A???B?WB"Q)B?A???BuB/??B E?BQ?BU?B??B΃BYsB	?oB?aB??B	?BB#_<B$??B??B-?B?LB=?B??B%I?B?cB??B"?B??BMB=`BrB
?jB??B?!BB*?BqB?4B?DBtBN?B?B?/B]?B??B??B5kB
?<B?lB ?tBX BôB??A?T
A??B?	B"@ B??A???B?B/?B @B{?B	B??B?TBzWB	?nB<?B3?B??B,`B#??B%6?B<yB-8B٣B;?B??B%@?B?B??BAmBN?B<yBPbB?^B
??BüB?GB??B9?B@>B?
B??B?9BC?B?}B ?BC?A?JdA؈fA?rA?~?AҮ?A?WA??Ag3?An??Az?iA???A??A/??A?a?A?hhA@?]Ak??@j?B??A??VA??*AlQ$AR0CAw?A???@?/?A<h?B?p@???@?J?@???A??A??PAڙ^A???AC?)wA???AZ?Agu?@B??????A?:?A??'A?[{@?<?A?PAŇB????B&?@???A?L@??C??!C?B 	?A??A؁?A?A?~?A??A??A??Af?^Aq??Az??A?tA?A/?A??BA??A@NAl?=@d:?B??A?f?A???Al?ARJOAw?A??@???A<?LB?n@?vm@??9@???AtA??AڀDA?=?Ai?C?(?A??AX??Ah??@C?~???mA??A??PA?q?@??XA?~?AĤ??xB?,@?'?A?}?@?Վ=	J?C??B )?                                 ,      H   )                  :      
               '            &      G   ?   ?   #      I         j   &               *      .   	   1      *   :   2   %                           '      %      1   '                  #                  '   #                  )   9   -         )         #   !                                                                     %      #                                             !                     %      '                     !                                          NЖ+N???NJ?8Ns?YN???O?^?Oi}ZO,??O?5?NM?O?f?N-TO??:OҖO.?dN8?O%7?O2??O?%O?t?Ot??O??OOZ?N?ZO?Ou*?O??N?RiN?B!N+?nO?u(O$?]P??O??P
??O--?N?i<O??N?GO?uO?ϘO?ǄNs?N/.O=Y[N?_?O?BbO,\?O&f?N??O@??N??O_GwOuYO)}O7??  h  ?  ?  ?  ?  $  ?  ?  4  ?  ?  ?  &  b  ?  ?  0    ?  ?  7  x    ?  ?    ?  ?  q  ?  ?  G  
`  ?    ?  ?  U    ?  B  ?  Z  /  w  ?  	/  o  
)  ?  	?  ?  	F    ?  ѽo??/?e`B?49X<t?:?o;o<49X<e`B<u<??h<?1=Y?=?P<???<???<?/<?`B<??h='??=t?<???=o=+=t?='??=t?=?P=t?=\)=#?
=<j=Y?>?w=H?9=e`B=H?9=?-=P?`=P?`=?9X=P?`=ix?=m?h=?%=q??=u=u=?%=?o=?O?=?+=??-=?E?=??=?S??????????????????????????????????????GKOSY[bhkhf[WOGGGGGGrtx?????????xtrrrrrr-+(-/<AHNUOH><:/----yvw|~?????????????~y???????????????????)45?BDEB65))6Bhmvyvvzt^ZOF6#ST`amnzzzymaaTSSSSSS;9;;9;HafltsjieaUH@;????????????????????????????????????????-./7<HUanstpaUHB<1/-"$/;HQHFCD=;.'")),)*56?CHGCA6*????????????????????????????????????????????????????????????????????????????????????????????????????'%&),6BOW[_`_^[OF9)'fghty???~tpgffffffff????????????????????  )6BF96)"$&/<HUajmmfUHC</&#"!!#/28<<<;7///#!!?????????	????????


#,.#










LMLOYh??????trhh[ROL????????
????????????)/9>;5)???'(*5B\gputrge[NB51*'?????)5>B@<5)??
#07<DA=60#
????
"
 ???????????
%&##
??	
#$#$#"
 	#)/2665/#
????????????????????????)389?)????????????????????????}??????????}}}}}}}}??????????????????=<?BHO[\][WOLB======????????????????????????????????????????UPQ[[ht????????zth[U11358?BNQUSONFB51111?????????????????????????????????????????????????????????????????????????????????????
"#(*%#
 ???0572/$#
	 
"#0?{ŇŔŠŭŲŲŭŠŔŌŇŀ?{?q?s?{?{?{?{?O?O?O?N?O?X?O?B?6?*?6?9?B?G?O?O?O?O?O?O?y?????????????????????{?y?s?y?y?y?y?y?yÇÏÓØÕÓÊÇÄ?z?w?x?z?~ÇÇÇÇÇÇ??????????
????????????????????????[?h?tčĚĢĥĤĚā?t?[?I?E?B?>?B?E?O?[?????????????????????????????????????????G?T?`?m?o?r?r?m?l?`?Z?T?G?C?;?:?7?;?C?G???????????????????y?m?`?P?G?G?P?m???????ѿտݿݿ޿ݿҿѿɿĿ??ĿĿпѿѿѿѿѿ?????????#?8???>?<?0?#?????????Ŀ???????/?<?C?F?<?/?#?"?#?,?/?/?/?/?/?/?/?/?/?/????????&?*?(?#???????ݽнǽ̽ҽݽ??(?5?A?N?R?X?\?X?N?A?5?(??
????? ?(?T?]?a?l?m?q?s?q?m?[?T?H?;?/?,?/?;?=?G?T?f?l?j?g?f?Z?W?U?Z?^?f?f?f?f?f?f?f?f?f?f?`?m?y???????????????y?m?c?`?V?O?P?T?\?`?F?9?<?:?8?-?!??????????????-?:?F?F?\?h?u?w?~Ɓ?~?u?o?h?\?O?C?=?=?@?K?O?V?\????????!????????????ŽŹż????????(?5?>?L?T?W?N?D?5?(??????????m?n?y?????????????????y?m?`?]?V?X?`?b?m?ʾ׾????????????????׾ʾ????????????Ⱦʿ????ĿǿĿ??????????????????????????????N?U?Z?^?a?g?i?f?Z?N?C?A?5?0?1?5?7?A?I?N?:?F?_?x?????x?l?f?_?V?H?:?-?'?%?"? ?-?:?(?M?Z?w??|?y?s?f?Z?M?A?+??????"?(?=?I?V?b?o?s?u?o?b?X?V?M?I?=?0?0?0?5?=?=???ûл׻ܻ??ܻһлû??????????????????????????????????y????????????????????????r???????????????????r?g?f?Y?X?O?Y?`?r?????????????????}?y?l?c?`?\?Y?\?g?l?y???#?0?<?I?N?S?R?M?=?0????????????????
?#?[?h?tĂćā?{?t?h?[?O?8?6?0?/?4?=?B?O?[²¿??????????????²?t?g?_?d?t²?ּ??????????
?????????ּʼüɼʼӼ?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D??N?[?g?t?v?t?g?[?N?M?B?=?=?B?E?N?????	???"?"?"???	???????????????????T?`?m?y?y?x?q?m?e?`?T?G?;?1?0?;?@?G?O?T?ɺֺ????????????????ɺº????????ºɺ????????????????~?r?e?Y?E?@?A?9?@?Y?~?????
???????
????????????????????????	????	???????????????????????????????	????!?$?"? ??	??????????????????ܻ????????	????????߻ܻӻܻܻܻܻܻ?ù????????????????????ùìÓÈÊÓÜêù?H?U?a?zÅ?z?y?n?`?U?H?<?/?,?/?0?6?<?C?H?3?@?L?Y?Y?e?m?m?e?b?Y?L?H?@?7?3?1?.?0?3?0?=?I?O?V?b?h?e?b?V?I?=?<?0?)?$?0?0?0?0?????ʼּݼݼټּͼʼ????????????????????	???????	? ???????????	?	?	?	?	?	?4?@?M?Z?f?n?f?\?M?G?'?????????4F1F=FJFVFcFpFwFqFcFVFJF=F$FFFFFF+F1E?E?E?E?E?E?E?E?E?E?E?E?EuEtEoEqEuEwE}E??B?6?*?????
????*?6?C?O?T?[?O?N?B F ^ a B ) 2 . - R S [ H   : w  ( (  ! ) 8 : 2 S 5 I C X ; G J  V ) 3 % B M ! 2 [ 0 _ Z O G ? [ 1 @ T = - %  ?  r  ?  ?  ?  V  ?  v    ?  ?  2  	  ?  |  k  d  ?  ?  ?  ?  M  ?  4  (    ?  ?  ?  h  ?  ?  }  O  ?  y  ?  G  ?  W  ?  o  ?  M  ?  ?  b  ?  t  ?  ?  ?    ?  n  ?  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  A^  f  f  g  f  `  Z  Q  F  ;  .  !      ?  ?  ?  ?  ?    g  ?  ?  ?  ?  ?    y  {  }  f  J  .    ?  ?  ?  ?  v  T  2  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  S    ?  ?  o  4  ?  ?  ?  n  <    ?  ?  _  ?  ?    %  4  >  =  4  $    ?  ?  {  7  ?  ?  U  ?  ?  $  "      ?  ?  ?  g  ,  ?  ?  W  ?  ?  '  ?  o  ?  D  1  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  j  M  -    ?  ?  ?  ?  }  p  ?  ?  ?  ?  ?  ?  ?  r  Y  E  2    
  ?  ?  ?  H  ?  ?    &  .  2  2  +    	  ?  ?  ?  ?  ?  ?  t  @    ?  ?  c    ?  ?  ?  ?  y  q  i  a  X  P  G  ?  6  ,  !          ?   ?  9  b  ?  ?  ?  ?  ?  ?  `  1    ?  ?  _  	  ?  ?  g  ?  e  ?  ?  ?  ?  ?  ?  ?  ?  {  m  T  0    ?  ?  ?  ?  i  G  %  Q  ?  ?  ?  ?  ?  ?    &      ?  ?  ?  k  4  ?      ?  
  =  M  X  Y  ^  a  `  X  J  1    ?  ?  k     ?  ?      ?  ?  ?  ?  ?  ?  |  ?  ?  ?  ?  ?  ?  ?  \  +  ?  ?  n  '   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  0  "      ?  ?  ?  ?  ?  ?  j  P  6    ?  ?  ?  ?  H      ?  ?  ?  ?  ?  ?  ?  ?  ?  u  [  8  
  ?  ?  ?  b  ?  R  ?  ?  ?  ?  ?  ?  ?  ?  r  V  2    ?  ?  ?  h  >    ?  ?  ?  &  W  y  ?  ?  r  ^  E  %  ?  ?  ?  p  $  ?  F  ?    N      )  0  4  7  6  0  %      ?  ?  ?  Y    ?  N  ?   ?  x  p  g  \  Q  B  2    	  ?  ?  ?  ?  ?  ?  ?  v  ^  1      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  l  M  +    ?  ?  ?  X  ?  ?  ?  ?  ?  ?  ?  ?  {  g  N  0    ?  ?  ?  ?  S  &   ?  9  ?  ?  ?  ?  ?    q  ^  A    ?  ?  ?  \  &  ?  ?  ?  ?     ?  ?  ?  ?  ?  	  ?  ?  ?  ?  u  :  ?  ?  m  %  ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  q  Z  4  
  ?  ?  ?  S    ?  !  ?  ?  ?  ?  ?  ?  ?  ?  {  n  _  N  :  $    ?  ?  ?  u  7  ?  ?  "  h  k  d  [  P  B  1      ?  ?  ?  w  A    ?  ?  V  )  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  u  n  h  _  N  <  +  ?  ?  ?  y  U  (  ?  ?  ?  @  ?  ?  Q  ?  ?    ?    ?  ?  :  A  E  G  F  C  <  0      ?  ?  ?  ?  f  C    ?  ?  _  
  
X  
`  
[  
I  
.  
  	?  	?  	e  	  ?    ?    ?  ?  ?  ?  ?    Y  .  *  ?  ?  Y  =  ?  ?  ?  ?  T  ?  /  4  ?  ?  r  M      ?    ?  ?  ?  ?  ?  ?  ?  _     ?  n  ?  ~  ?  v  l  d  ?  ?  ?  ?  ?  ?  ?  ?  ?    i  O  +  ?  ?  ?  4  ?  n  ?  ?  g  D    ?  ?  ?  k  :    ?  o  !  ?  ~  3  ?  ?    Q  #  "  :  `  ?  ?  	    /  U  B    ?  a  ?      ?  U    ?  ?  ?  ?  ?  ?  u  [  C  ,    ?  ?  ?  ?  ?  ?  x  Q  ?  v  f  T  @  *    ?  ?  ?  ?  q  >    ?  |  5  ?  M  o  
  
?  :  ?  ?    ;  @  .    ?  ?  3  
?  
5  	V  8  ?  J  ?  ?  ?  ?  ?  ?  ?  ?  {  \  5  
  ?  ?  ?  d    ?  k    F  Z  N  B  >  6  $    ?  ?  ?  ?  p  =    ?  ?  Z     ?   ?  /  )  #                 ?  ?  ?  ?  ?  ?  ?          K  c  q  w  r  g  X  C  (    ?  ?  \    ?  Y  ?  ?  ?  ?  ?  v  _  E  '    ?  ?  ?  ?  z  e  Q  I  y  ?  ^  7    	/  	  ?  ?  ?  u  >    ?  u  "  ?  e  ?  T  ?  ?  <  ?  ?  o  Q  2    ?  ?  ?  ?  ?  e  A    ?  ?  S  ?  ?    ?  (  
(  
(  
#  
  
  	?  	?  	?  	O  	  ?  ?  X    ?  w     ?  ?  ?  ?  {  u  l  b  W  H  :  3  .  #      ?  ?  ?  ?  ?  ?  ?  	u  	?  	?  	?  	?  	y  	\  	8  	  ?  ?  F  ?  ?    ?  ?  >    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  	3  	D  	B  	3  	  ?  ?  =  ?  h  ?    ?  ?  {  ?  ]  7        
?  
?  
?  
?  
?  
K  	?  	?  	h  	  ?  c  ?  
  ?  ?     8  ?  ?  ?  ?  ?  p  G    
?  
?  
.  	?  	o  		  ?  )  ?  $  ?  ?  ?  ?  ?  M  1    ?  ?  ?  v  9  ?  ?  ?  ?  8  z  ?  ?  ?