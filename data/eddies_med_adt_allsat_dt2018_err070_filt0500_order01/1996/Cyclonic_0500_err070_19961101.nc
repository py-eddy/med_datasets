CDF       
      obs    C   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ????         	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?`bM????   max       ???t?j~?       ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M?Յ   max       P???       ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ?ě?   max       <T??       ?   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?(?\)   max       @F?z?G?     
x   ?   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??z?G?    max       @v?z?G?     
x  +H   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @         max       @Q`           ?  5?   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @??        max       @?V@           6H   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ?1'   max       ;?`B       7T   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A?uv   max       B4?q       8`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?|   max       B4?O       9l   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >yqh   max       C???       :x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >Zޢ   max       C???       ;?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          `       <?   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9       =?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -       >?   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M?Յ   max       P,?       ??   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ????+j??   max       ???D??*       @?   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ?ě?   max       <#?
       A?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?+??Q?   max       @F?z?G?     
x  B?   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??z?G?    max       @v?          
x  MP   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @         max       @Q`           ?  W?   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @??        max       @?X?           XP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B?   max         B?       Y\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ?~?Q???   max       ???D??*     ?  Zh   `         5               2                   	      ^   0               	         .   "   $         ?   	   +         ,         5                  
       %         '            
            ?                     8   	   %P???N3?N?9eO??NBPO???O/?=N?y?O???P??Or^wNBNg?fM???M?ՅN.??PQ?HO?K6NͺN?"HO$?GN?o?Nܣ?N?ŐNB?P2?]P??PH?Ng?vN???O?W	O7??Po?N?k?ON/?P2?O?\O@\P??N?<?OM?~OF??O?AN???N???O?+?O???Oi?'N? XOD1/O??N?n>N?p?N?F?N??=N??N???O???O=?:O??O??mN?AGN8.=Oj?[Pq?O̄N?R?<T??<#?
;?`B;?o?ě??ě??o?o?t??D???D???T???T???T????C???t????㼣?
???
??9X??9X??j?ě??ě????ͼ?`B??`B???+?C??C??\)??P?????w?#?
?,1?0 Ž0 Ž0 ŽD???D???D???H?9?P?`?P?`?]/?]/?m?h?m?h?u?u?}󶽃o??o????????O߽?hs??hs??t????P???P?????????????ě??????
*+"?????????)5851)???????????????????????#/<@FGEA</#
??? 
#'##&#
      #,6COTVTQC*#/<@HH@<:/*'#

#0;830#






????????????????????????????????),6<LU\`b[OB6)&)5BBB>54)$$&&&&&&&&????????????????????????????????????????????????????????????OOP[[]hljh[OOOOOOOOO]ht??????????|}h[]????!%*.0)?????%)158ABFGCB5)'  %%%%????????????????????????????????????????????????????~????????')/)???????05BN[egrge[NEB;50000????????????????????,7?HSTam????zmaTH;.,TWW[clw????????shaTTP]\ez?????????n^UMHP????????????????????ABKORX[\^_]\[OLB>;=A/2<IUbqyzibU<68>?<1/X^chkt{???????th[XQX$0BN[g???????g[B)$15BGNNPPNB5531111111????????????????????)Rltz?????tb[J5(!)????????????????????),5?BKB?5)??)1,&%)45)????=BEO[`hhhh^[OB??====?????????????????????????
#.#
??????3>Ibn}??????nbU<10:3GILMQUYbnvyxsnb]UIGG????????????????????Ua^gnt?????????ztf[U#)2<H`jnzzna</#w|{?????????????{w????????????????????????????????????????sy????????????????ts????

??????????????  ??????????&)35BNPXNLGB54,)($&&^glt?????????tpg^^^^mmqzzz???????zmmmmmm????????????????????
#<HTTQSOF</#?????????????????????Ua???????????zunaURU?????????????????????????????????????NU[gt|??????xg[XTNMN?????????????????????????????????????????????????~{}?????????F?1?(??5?5?N?s?????????????????????g?F?g?\?]?_?g?n?s?w?y?s?g?g?g?g?g?g?g?g?g?g?
??????????
??#?%?+?/?:?/?#??
?
?
?
??????ŹŴŲŲŷŹ????????????????????????????????????????????? ???????????????.?"????????׾ϾȾ̾׾????	??,?3???;?.?)?"?!?#?'?)?6?B?[?g?h?p?h?[?W?P?O?B?6?)?!?????????!?&?"?!?%?!?!?!?!?!?!?n?a?X?V?^?n?zÇÓàïù??øìàÓÇ?z?n?/?"???????????????"?/?H?T?a?Y?]?T?H?/?:?2?*?-?1?:?F?S?_?l?x??????x?s?p?l?S?:?m?k?c?m?m?n?z?????????z?m?m?m?m?m?m?m?m?f?c?Y?U?M?H?K?M?R?Y?f?j?r?t?r?r?f?f?f?f???????????ʾѾ̾ʾ???????????????????????????????????????????????????????????ϹϹù????????ùϹֹ۹ϹϹϹϹϹϹϹϹϺ?????.?6?L?r?????????????r?Y?3?+??h?a?[?O?F???F?O?[?tāčėĘĖčąā?t?h?5?.?)??)?5?7?B?N?[?]?a?]?[?N?B?5?5?5?5?????????????????? ???????àØÓÒÉÍÓàèìóù??????????ùìà??????????????? ???????????????????޿m?k?b?e?m?r?y?|???????????????y?m?m?m?m???????????????????????????????????????????????????????ûɻû????????????????????	??????????????????????;?I?U?Q?B?/??	?s?Z?A?(???????ݿ?????(?A?N?g???????s?m?a?T?;???/?H?a?m?w???????????????z?m?U?P?U?W?a?c?n?v?u?s?n?a?U?U?U?U?U?U?U?U?T?N?G?;?G?T?[?`?m?y?????????????y?m?`?T??????|?|???????о??)?:?4?????Ľ????Ľ??????????????????Ľнݽ????????ݽнľ????վѾӾھ????	?"?/?<?<?8?2?"??	???????????????????????????????????????????ſ??ѿ????????????Ŀѿݿ????????????\?s?????????????	????????????????s?\?????????????*?6?@?O?h?l?\?W???6?*???ŠŝŔőőœŔřŠŧŭŸŹżſŽŻŹŭŠ?Y?4?'? ??"?4?M?Y?f?}???????????????r?Y??y?r?j?k?r????????????????????????W?Z?c?r?~?????????ɺԺɺ????????~?r?e?W?:?6?-?+?*?.?:?F?U?_?e?b?l?n?l?f?[?S?F?:?:?-?'?$?'?-?:?_?z????????????p?_?S?F?:???y?s?g?Z?R?N?H?D?N?Z?g?s?x?~??????????ƚƐƎƈƎƑƚƧƳƸ????????????ƳƧƚƚ??????žŭŨŐŇśŭ?????????????????E?E?E?E?E?FFF$F,F:F=FHFSFSFHF@F*E?E?E??ܹϹù??????¹Ϲܹ???????%?$?????ܻ??????!?-?:?F?S?U?S?O?F?:?-?!???r?l?`?l?x???????????ûƻǻ????????????r??????????????????????????	?? ?!???????????????????????????????????????????????????????$?0?4?5?0?+?$???????????"?#?/?8?<?H?T?L?H?<?2?/?#???/?)?#???#?-?/?<?H?K?H?D?=?<?3?/?/?/?/?????????????????????????
?????????????????????????????????????????????????????E*E%EEE EEEE*E7ECEPEVEYE[EWEPECE7E*?!????	???!?.?:?G?R?^?Y?S?G?:?2?.?!???y?s???????Ľݽ????????нĽ??????????H?A?M?Z?s????????׾??ܾʾ????????l?Z?H?=?9?0?$? ?$?&?/?0?=?I?L?V?Y?[?V?K?I?=?=??????????????????????????????????????ľĳĮīĭĳĿ????????????????????????ľ??????ĿĹīĩĿ?????????#?0?:?0?#?
?񼽼????ʼּ???????????
? ???????ּʼ???????????????????????????ùõõù?????? % = < < d f 1 _ ; E I K h T : % g . - % K ) X ^ @ : } 1 R V j ( J F ` | d P @  n E ; M O = r ] ? S z V # C ] > 8 * ) s x  d  5 v     ?  T  ?  c  ?  ?  ~  ?    ?    [  ?    
  M  T  =  ?  ?  x  ?  *  +  U  .  ?  s  ?  C  ?  ?  ?  ?    |  \  f  _  ?    ?  	        F      ?  l  ?  ?    ?  ?    J  ?  ?  D  ?  z  ?  w  ?  ?????;?`B?t??D???D???ě???/?D???q????w?0 żu?????u???ͼ?`B??`B??C??C????h?#?
?t??+??????P???P?}󶽇+?<j?'ě??49X???
?H?9?L?ͽ????y?#?e`B?\?aG??????C???t??]/?u???罺^5??C??}???ě????㽅??????????????-??+???罩???ȴ9??{???
???;???{?1'BP?B??BDcB?_B#?B0#?B,dB% ?B?aB_$B?B$?B ?JB4?qB??BPyB ?xB?BB?B$?B9QB??BA?B SNA?uvB pB<?B"?B??B&?B$?B	;B?yB*?BhB??Bq?B
?B??BM?B#?%B'ފB'??BESB?B?B?B?HBB!?Bd?B??Bv?B	?dB B??B?B?oB?BHBLB
??B	h9BA`B,?FB
?B??BG?B>B??B*)B0@?B)?B%S;B??Bs?B??B)B!:EB4?OB?BD?B }?B?B?B0?B?B=?B??BX/B A?A?|B ??B?B"*?B?^B'@8BFB	?tB??B*?$B??B;B	?B?_B68B@B#?lB'??B'?*B@?B@BA?B9	B?:B>?B?/B??BE?B?uB
?A???B??B??BwB?*B?B
??B:?B	??B\lB,?8B8?A???A??A??=A???A???AY
?A؋?Aj?AɏA?tO@?t?A?W?@?2?AO8OA?Vo>yqh??b3A?IA??HA1҇A???A?6\Am?-Aq??@??A??tA?+A?SWAƯ?AkD?A)K?A'6
AZ??A???A}C?A?f?A?kA???@?m@??$@??@?	:@??A???B?qA??\C????Xq@s?@??A?;?A?4?B	Y?A?p?A??A殏A?8?C???A?A%?@AG?HB
ӰB??A???A???A?5A?qA?A???A?k>A?~A?zwAXлA׉?A
??A?dA?8?@???A?*?@??AN??A?}?>Zޢ??8?A?ҹA?o=A1?lA?b?A?EAm?As@?AA???A???A??HAư?Ak??A%
CA$D?AZ??A???A|92A???A?@pA??@??@?????H?@?#@?A??B?rA?{BC????!],@tc@??bA??A??B	<KA?s?AÀ?A?O?A???C???A?<A$5?AA?B4?B??A?rOA??]A?FA?x
   `         5               3                   	      ^   1               
         /   #   %         ?   
   ,         ,         5                  
   !   &         '            
            @                     8   
   &   9               #            )                     7                           -   1   /         /      '         3         '            %         '   %            '                              '            '         #                           )                                                +   -   -         )               '         #            %            #            '                              '                  Pt?N3?N?9eO@?NBPO"?`O/?=N?y?OA|P??N??@NBN3 M???M?ՅN.??O5=?O.)?N?)?N?"HO	?NN?Nܣ?N?ŐNB?P,?P ?P%??Ng?vN???O?<YO7??Ow{zN?k?ON/?O???O?\N??O??N?n(OM?~O?O˛?N???N?u?Ot?O?`?Oi?'N? XOL?O??N?n>N?p?N?F?N??=N??`N??aOb?OO=?:O??O??mN?AGN8.=OZRLO?=?O̄N?R?    !  ?  ?  ?    ?  ?  ?    4    p  5  M  ,  	[  	  ?  C  ?  ?  ?  ?    >  ?  +  /  ?  ?  ?  Y  4  U  Y  ?  D  ?  ?  '  R  ?  ?  Y  ?  ?  e  T  v  ?  	  }  =  ?  D  ?  F  ?  t  ?  ?  ?  6  #  <  ????h<#?
;?`B?o?ě??49X?o?o?ě??D???ě??T???e`B?T????C???t???hs?+??j??9X?ě???/?ě??ě????ͼ??h???h?C??+?\)??w?\)?aG??????w?T???,1?8Q??P?`?49X?D???P?`?H?9?H?9?Y??q???e`B?]/?m?h??+?u?u?}󶽃o??o??+??7L??????hs??hs??t????P???P???㽸Q콛???ě???????	????????)5851)????????????????????? 
#/7<@@>8/#
? 
#'##&#
      $*46CKPONJC6*"#/<@HH@<:/*'#

#0;830#






????????????????????????????????$).6<BLOUXTOB@6))#$$&)5BBB>54)$$&&&&&&&&????????????????????????????????????????????????????????????OOP[[]hljh[OOOOOOOOO??????????????????????!#%)$
????()5;BBBA<5-)""((((((??????????????????????????????????????????????????????????????')/)???????05BN[egrge[NEB;50000????????????????????-07?HMTm????zmaTH;.-UXX\dlx???????{rfaTUYedjz??????????zn\WY????????????????????BBOTZ[]]\[OEB?<>BBBB7IUbovxtgbUI=;@A=327X^chkt{???????th[XQX=EN[gt??????tg[NHB?=15BGNNPPNB5531111111????????????????????')6BO^lry??xsh[QH6-'????????????????????()5<B51)????&('!"(!?????@BGO[_gg][ODCB@@@@@@????????????????????????? 
??????6?Ibn|?????nb^I<34<6GILMQUYbnvyxsnb]UIGG????????????????????y??????????????~xuty#&0<H^hnznaUH</#w|{?????????????{w????????????????????????????????????????sy????????????????ts????

??????????????  ??????????&)35BNPXNLGB54,)($&&^glt?????????tpg^^^^rz???????znnrrrrrrrr????????????????????	
#)/<BHJLH=/#
	?????????????????????Ua???????????zunaURU?????????????????????????????????????NV[gt{??????ug[XUNMN?????????????????????????????????????????????????~{}?????????g?Z?R?P?Q?X?g?s?????????????????????s?g?g?\?]?_?g?n?s?w?y?s?g?g?g?g?g?g?g?g?g?g?
??????????
??#?%?+?/?:?/?#??
?
?
?
????ſŹŸŶŶŹſ????????????????????????????????????????????? ?????????????????????׾վо׾??????	???!?"?#?"??	???)?"?!?#?'?)?6?B?[?g?h?p?h?[?W?P?O?B?6?)?!?????????!?&?"?!?%?!?!?!?!?!?!?z?x?n?d?b?j?n?zÇÓàâìðìâàÓÇ?z?/?"???????????????"?/?H?T?a?Y?]?T?H?/?S?J?F?B?F?H?S?V?_?l?x?{?y?x?r?l?l?_?S?S?m?k?c?m?m?n?z?????????z?m?m?m?m?m?m?m?m?f?e?Y?V?M?J?M?Y?f?i?q?j?f?f?f?f?f?f?f?f???????????ʾѾ̾ʾ???????????????????????????????????????????????????????????ϹϹù????????ùϹֹ۹ϹϹϹϹϹϹϹϹϺ3?2?3?8?@?L?T?Y?e?r?r?w?u?r?l?e?Y?L?@?3?h?[?O?N?G?O?P?[?h?tāčđĐčĈā?z?t?h?5?3?-?5?A?B?G?N?[?^?[?Y?N?B?5?5?5?5?5?5?????????????????? ???????àÙÓÏËÏÓàäìðù??????????ùìà?????????????????????????????????????m?k?b?e?m?r?y?|???????????????y?m?m?m?m???????????????????????????????????????????????????????ûɻû????????????????????	??????????????????????:?H?S?P?@?.??	?s?Z?A?(???????޿?????(?A?Z?g???????s?m?a?T?;?#??/?7?H?T?a?p?????????????z?m?U?P?U?W?a?c?n?v?u?s?n?a?U?U?U?U?U?U?U?U?T?S?J?T?`?d?m?y???????????y?m?`?T?T?T?T???~????????нݾ??%?.?(?????Ľ??????Ľ??????????????????Ľнݽ????????ݽнľ????????????????	??"?)?*?(?#????	?????????????????????????????????????????ſ??ѿ????????????Ŀѿݿ??????????????t?n?r?z???????????????????????????????????????????*?6?@?O?h?l?\?W???6?*???ŔŒŒŔŔśŠŭŷŹżŻŹŹŭŠŔŔŔŔ?r?Y?M?7?,?%?#?'?4?M?Y?r??????????????r??|?r?l?l?r??????????????????????W?Z?c?r?~?????????ɺԺɺ????????~?r?e?W?F?:?0?-?-?-?2?:?F?S?[?_?l?l?h?b?_?W?S?F?:?-?'?%?(?-?:?_?x???????????u?n?_?S?H?:???y?s?g?Z?R?N?H?D?N?Z?g?s?x?~??????????ƚƗƎƋƎƕƚƧƳƳ??????ƳƳƧƚƚƚƚŭťŝŜŠŪ????????????????????????ŹŭE?E?E?E?E?E?FFF$F/F=FJFPFMFKFGF?F(E?E??ܹϹù??????¹Ϲܹ???????%?$?????ܻ??????!?-?:?F?S?U?S?O?F?:?-?!?????????y?~???????????????ûĻ?????????????????????????????????????	?? ?!???????????????????????????????????????????????????????$?0?4?5?0?+?$???????????"?#?/?8?<?H?T?L?H?<?2?/?#???/?)?#???#?-?/?<?H?K?H?D?=?<?3?/?/?/?/??????????????????????????????????????????????????????????????????????????????E*E)EEEEEEEE*E7ECEPEVEXEUEPECE7E*?!????	???!?.?:?G?R?^?Y?S?G?:?2?.?!???y?s???????Ľݽ????????нĽ??????????H?A?M?Z?s????????׾??ܾʾ????????l?Z?H?=?9?0?$? ?$?&?/?0?=?I?L?V?Y?[?V?K?I?=?=??????????????????????????????????????ĿĳįīĮĳĸĿ??????????????????????Ŀ??????????ķĹ????????????
????
???񼽼????ʼּ???????????
? ???????ּʼ???????????????????????????ùõõù?????? 1 = < + d G 1 _ 5 E C K e T : % 1 , & % P 8 X ^ @ 5 { 4 R L h ( @ F ` l d G B   n + < M 9 ) k ] ? E z V # C ] 1 3 % ) s x  d  ' v     l  T  ?  ?  ?  ?  ~  ?  [  ?    [  v    
  M  ?  p  ?  ?  N  a  *  +  U  ?  ?  ?  ?  ?  ?  ?     ?    $  \    ?  ?    C  ?    ?  ?  ?      +  l  ?  ?    ?  ?  ?  ?  ?  ?  D  ?  z  ?  V  ?  ?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?    ?    ^  ?  ?  ?  ?      ?  ?  ?    ?  ?  M  ?  ?  ?  !  &  +  1  6  ;  A  J  U  `  k  v  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  ]  B  %    ?  ?  ?  E  ?  ?  7  ?  =  ?  *  ?  ?  ?  ?  ?  ?  ?  o  1  ?  ?  N  ?  ?  9  ?  ?  ?  !  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?      ?  ?  ?  ?  ?  ?  ?  `  0  ?  ?      ?  ?  ?  ?  ?  ?  ?  ?  y  R    ?  ?  ?  ?  c  0  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  y  r  j  b  Z  R  D  4  $        j  ?  ?  ?  ?  ?  ?  ?  ?  u  :  ?  b  ?     ?  ?  S   ?      ?  ?  ?  ?  ?  b  5    ?  ?  |  E  
  ?  ?  n  F    ?  ?  ?      !  ,  4  3  '  	  ?  ?  ^    ?  ?  3  ?  ?          
         ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  \  c  i  o  p  o  o  i  a  Y  I  3    ?  ?  w  5  ?  ?  n  5  .  '              ?   ?   ?   ?   ?   ?   ?   ?   ?   o   ]   K  M  Q  T  P  H  ?  2  &      ?  ?  ?  ?  ?  ?  k  P  4    ,  /  0  ,  '  "          ?  ?  ?  ?  ?  ?  ?  ?  [  2    ?  ?  -  ?  ?  7  ?  ?  	.  	S  	V  	4  	  ?  S  ?  ?  1  ?  ?  ?  ?  ?  ?  	  	   ?  ?  ?  W    ?  w    ?  ?        a  |  ?  ?  ?  ?  ?  {  g  K  -    ?  ?  ?  U  ?  ?  ?   ?  C  A  ?  ;  3  +  #          ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  _  7  	  ?  ?  k  /  ?  ?  ?  l  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  `  &  ?  ?  O    ?  X   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  \  3  ?  ?  z  9   ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  }  b  G  +    ?  ?  ?  ?  v  i  \      ?  ?  ?  ?  ?  ?  j  T  @  ,      ?  ?  ?  ?  ?  ?  9  5      ?  ?  ?  ?  y  U  )    ?  ?  y  3  ?  ?    ?  ?  ?  ?  ?  c  =  ?  ?  _  5    ?  ?  G  ?  l  ?  =  _   m    $  +  (    
  ?  ?  ?  ?  ?  q  J  0  .  ?  ?  T  )  ?  /  &    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  z  l  ^  Q  L  [    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  q  X  8     ?   ?   ?  t  ?  ?  ?  r  \  _  c  >  	  ?  ?  6  ?  X  ?  ?  s  ?  /  ?  ?  ?  ?  o  Y  B  (    ?  ?  ?  ?  ?  k  S  2     ?   ?  ?    "  0  ;  E  O  V  Y  U  L  =  !  ?  ?  L  ?  U  w  J  4  $      ?  ?  ?  ?  ?  \    ?  ?  x  M  "  ?  ?  ?  ?  U  M  C  6  #    ?  ?  ?  ?  r  N  *    ?  ?  ?  w  K    ?      %  *  X  Q  9    ?  ?  ?  D    ?  r  ?  i  ?  H  ?  ?  ?  ?  ?  ?  ?  ?  ]  2    ?  ?  f  2    ?  ?  q    )  2  <  C  4  !    ?  ?  ?  v  H    ?  ?  ?  d  ,  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  >  ?  ?  n    ?  ?  G    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  k  J  $  ?  ?  7  '          ?  ?  ?  ?  ?  ?  i  0  ?  ?  v  -  ?  J   ?        O  ;  !    ?  ?  ?    `  C  !    ?  ?  ?    ?  ?  ?  ?  z  ]  ;    ?  ?  ?  ?  ]    ?  ?  V  ?  ?  5   ?  ?  ?  ?  ?  ?  ?  z  q  f  X  J  ;  *       ?   ?   ?   ?   ?    2  R  W  X  U  O  E  7  (      ?  ?  ?  ?  }  `  @     ?  V  ?  ?  ?  ?  ?  ?  ?  ?  w  W  4    ?  ?  ?  ?    ?  ?  ?  ?  ?  ?  ?  p  <     ?  ?  1  ?  c  ?  {    ?  |  ;  e  a  Z  P  D  6  %    ?  ?  ?  ?  ?  {  n  _  D    ?  ~  T  N  I  D  >  =  J  W  c  p  s  l  e  ^  W  8    ?  ?  ?  S  f  l  m  v  g  O  -  ?  ?  f    ?  e  *  ?  ?  h     ?  ?  o  u  g  J  ,    ?  ?  ?  ?  b  6  
  ?  ?  ?  `  .    	  ?  ?  ?  ?  ?  ?  ?  q  c  U  G  :  .  #          
  }  j  W  B  -    ?  ?  ?  ?  t  h  c  z  ?  ?  ?  ?  ?  ?  =  (      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  v  O  '  ?  ?  ?  ?  ?  ?  |  [  1    ?  ?  ?  u  j  W    ?  .  ?  A  C  G  M  E  :  /  %        ?  ?  ?  ?  ?  ?  ?  ;  ?  |  ?  ?  ?  p  [  C  )    ?  ?  ?  q  A    ?  ?  c  &  ?  ?    6  F  =  )    ?  ?  ?  {  7  
?  
v  	?  	'  T  s  n  ?  ?  ?  ?  ?  s  U  4       ?  ?  ?  ?  ?  t  U  5      ?  t  b  O  =  *      ?  ?  ?  ?  ?  ?  y  g  9  ?  ?  j    ?  u  D    ?  ?  ?  h  )  ?  ?  _    ?  ]  P  3  ?  p  ?  ?  ?  }  o  _  N  8  !    ?  ?  ?  ?  |  f  L  1    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ^  >  &    ?  $  2  !  
  ?  ?  ?  ?  U  %  ?  ?  g    ?  r  ?  v  ?  x  k  ?  ?  ?    !        ?  ?  ?  r  '  ?  b  ?  D  ^  ?  <  ,        ?  ?  ?  ?  ?  ?  ^  &  ?  ?  ?  ?  ?  a  ;  ?  ~  I    ?  ?  j    
?  
m  
  	?  	5  ?  \  ?  ?    ?  ?