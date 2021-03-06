CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?h?t?j~?   max       ??$?/??        ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M?ӷ   max       P?@?        ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ??9X   max       >$?        ?   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>???Q?   max       @E}p??
>     
    ?   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??z?G?    max       @vqp??
>     
   *?   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @'         max       @Q?           ?  4?   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @??        max       @??`            5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ???
   max       >J??        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A?   max       B+3?        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?}`   max       B*Ȋ        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =Y?   max       C?n?        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =|eM   max       C?w?        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M?ӷ   max       PqP        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ???*0U2b   max       ??ě??S?        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ??9X   max       >1'        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>???Q?   max       @E}p??
>     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??(?`    max       @vp?\)     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @'         max       @Q?           ?  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @??        max       @???            U?   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D?   max         D?        V?   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ???)^?	   max       ????7??4        W?                     H      	      W      (   h         I      ;   ]                        /               ?            	                  M               	            )   ?                     5         :   'M??aN2aN?=NV??N%??OFu?P?eN??N;??NC?mP??;NٳO???PEe?N?=jNR??Pp??O?sPT?rP?@?M??ONe??N?q?N?h O??O-?NNp?O?TNkixN?<OL?NR?~P??nNţ&N???O)9NĜ?Nj
N??>Ni7]O;?N?ܸP/bpM?ӷO?
O??O[c?N?`5NNO?@O???O?OڗN?E?N QANBAHM??N???N??O?W?N???N?4CO?{O7N??9X??t??#?
?#?
?o?o?o?o???
??o:?o;o;D??;?o;?o;??
;??
;??
;?`B<o<o<t?<t?<D??<D??<u<?C?<??
<?9X<?9X<???<???<???<???<?/<?`B<?`B<???=o=+=\)=t?=??=?w=?w=?w=#?
=#?
=D??=L??=T??=T??=Y?=Y?=]/=u=?%=?o=?C?=???=??
=??=??>$?????????????????????CNQ[gmig[NCCCCCCCCCCww??????????????wwww-$$)/<?B?<1/--------ONBABDO[a^[OOOOOOOOO???????????????????????????????????????????????????????????????????????????????????????????????????? &+HN????????naU<$),-*)
#/BRSLKE</#
????????
??????63<<<HUX_WUHA<666666?)0575)?????????? &/<ntmaH<????ynoz}????????????zyy????
#/;RYYOKH<????!Ni?????????tc]MBzxvz~????zzzzzzzzzz"+/;<C;/""????????????????????')+16:BGFBA6,)yx}z|?????????????//9;AHTaba]WUTLH;3//feehqtuzzthffffffff????????

????????????????????????????????????TMKTaemqz??}zsma_TTTmmoz{?????zxpmmmmmmm??????"5N[g?~[C5???%&/4<HJUX^[UIH</%%%%ww????????????????ww??????????

??????????????

?????,,/<?HIH</,,,,,,,,,,???????	?????????????????????????????????????????????????????????????????????????????????????????????????????-()/<@HQRUaa^UTH<4/-????????????????????????????????????????		
#0:<B:0-#
	TZ[ahqskh[TTTTTTTTTT)BUZRKB5)
6;HLOX[^[H;/" &(6(%&,/3<HNSUSNH</((((????????????????????

????????????????????????????W[gptvttgd\[WWWWWWWW
?????
zww{}?????????|{zzzz????????????????????? ).9@A=61)
?
 #//430/(#
?????????????????????????????????????????njinoz??????????znnn??????????????r?l?r????????????????????????????????????????????????????????????[?g?p?t?t?t?t?k?g?[?Q?P?N?M?N?T?[?[?[?[?????????????????????????????????????????l?l?t?y???????????y?w?l?l?l?l?l?l?l?l?l??*?.?6?C?L?H?>?6?*??
????????????????????	?????????r?h?q?????????????????????????????????????????????????????????????????????????????????????????Ҿ(?4?8?A?F?A?:?4?(?'??%?(?(?(?(?(?(?(?(??(?N?Z?????o?V?Q?A?(??????????ٿԿ?????????????????????????????????????`?m?y?????????????y?m?`?T?H?@?4?5?;?M?`????)?6?>?B?G?D?1?)???????????????????E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?Eپf?i?i?m?o?f?^?Z?W?U?V?Z?`?d?f?f?f?f?f?f???׾????	???ھо¾???{?????s?k?o????E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E??;?H?T?d?i?e?[?H?/??	??????????????)?;????4?;?R?H?B?"???????g?5?=?g???????????Y?e?r?z?~???~?r?i?e?Y?X?Y?Y?Y?Y?Y?Y?Y?Y?T?a?a?g?m?o?o?m?a?W?U?T?T?S?T?T?T?T?T?T?????????????????????????|?|????????????ÇÓÙÓÑÓàâàÛÓÇ?|?z?q?z?~ÄÇÇ???????'?4?7?:?4???????????׻λɻл??"?#?/?2?;?@?C?<?;?:?/?"???	???	??"?A?M?Z?`?f?h?f?Z?M?A?7?8?A?A?A?A?A?A?A?AD?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D??ûлٻܻ????????ܻһлû????ûûûûûû???????????????????????āčĚĞĦĦħĦĤĚčā?|?t?m?s?tāāā??????????????????????????????????????????)?:?B?[?mā?z?e?????????????????n?z?ÇÍÓÓÞÓÇ?z?s?n?m?i?l?n?n?n?n?????¹ù͹ϹٹܹݹܹڹϹù????????????????????????????????????????s?h?n?r?|???'?4?@?D?M?Y?\?Y?O?M?@?4?'?&?????¤¡???ʾ׾?????????????????׾ʾǾ?????????ÇÓàáåàÕÓÍÇ?~?ÇÇÇÇÇÇÇÇ?f?h?s?w?u?w?~?s?f?a?Z?M?A?9?5?A?B?M?Z?f?!?-?:?:?A?>?:?-?+?!?????!?!?!?!?!?!?g?????????????????????s?Z?N?C?????F?Z?g?
?	????
??????
?
?
?
?
?
?
?
?
?
????????!??????????????????????"?.?;?G?T?X?U?T?R?G?;?.?"??????"?"???????ݿ???????????ݿο??????????????f?s?????????????????????s?k?f?[?Z?d?f?:?F?G?S?X?S?F?:?8?9?:?:?:?:?:?:?:?:?:?:????????????y?m?T?G?=?9?7?<?G?T?`?m?y???b?g?p?n?b?I?<?#????????????
??0?<?I?bD?EEEE&E*E6E0E*EEED?D?D?D?D?D?D?D??b?n?zÇÓÖÑÃ?n?a?U?H?C?6?2?3?:?H?U?b????????????????????????żž???????????ƽ????????????????????????????????????????{Ǉǃ?{?w?o?b?b?_?b?o?t?{?{?{?{?{?{?{?{āĀ?t?q?h?^?h?tāăĂāāāāāāāāā?нݽ????????
????????ݽֽнϽннн?¦²¿????¿²¦£¥¦¦¦¦¦¦¦¦¦¦???????ֺ׺Ӻͺɺ??????????~?x?v?z?~????ǭǰǲǭǫǡǠǔǈ?{?z?{?|Ǆǈǒǔǝǡǭ?.?:?C?G?M?G?@?:?.?)?!??!?$?.?.?.?.?.?.?ʼּ??????????????ּʼ????????????ʻ??????????????????????????????????????? m  O / 5  L ? R . 4 e $ - F ? T + U e x q , X L S D ' S 5 + i D f u Z > L W 8 2 ? > ` & : { E h A J &  K X L { n 3 ! T ' L &  (  ?  ?  q  A  ?  ?  }  u  _  ?  1  v  ^  ?      I  ?  ?  N  ?  ?  ?  ?  6  f  K  ?  3  )  x  n    ?  ?  ?  ?    ?  P  ?      +  %  ?  &  J  ?  ?  A  ?    W  f  ]  ?  2    ,  ?  ?  P???
?e`B%   ?ě???o;?`B=?o??o;D??;??
=?9X;??
=0 ?=?"?<#?
;?`B=??w=o=??=??<D??<T??<?9X<???=0 ?<?1<?j=?+=+<???=?w=o>J??=??=0 ?=,1=?P=?w=0 ?=<j=}??=L??=?x?=0 ?=y?#=49X=?+=H?9=T??=???=??
=?v?=??=?t?=ix?=??=?O?=???=??P>$?=??=?/>??>-VB)??B?ZB
?8B^?B?8B??B??B??B?B"3jB??B??B??B<B>CB
?B?[BAB?JB	3B{?A?B".6B?	B??A???B[B?B?B??A?A??:B??B#WB">B#z?B#??B?JB?B"
WBP?B??B?sBGB?_B!?%B?B%.?B??B?,A??xB?B?qBTFB+3?B	4~B?!B)XBmZB??BYB??B?B?TB)??B?`B7BBF?B?B??B??B?RB?[B"@?BC?BƉB??BA?B??B}?B??B=#B??B	?dB?A?}`B"A3B??BGfA?EPB:?B<?B??B??A??
A?zB?B6?B>?B#?%B#?"B??B,B"@B@B??B?-B>?B??B!??B?B%$B?gB?A?nCB?$B7YB?sB*ȊB	A?B>?B)C?BA?BBC?B??BD}B?"@?/RA???A???A?FA??A?</A???A?b0A?,0A8/?A??A?}?Aj?rA?eC?n?A@?AOΟC?c^A?NA??O??G?A?6?@??A??@???A??A=5C??@?6^@\~?Aރ?B?RA? ?AȨ?=Y?@??^@?A???AS?YA??A?5O@q2?A?[A??1A?;_Ab'?Az??AC??@?P?Ak
?A?_iC?eEAƲ A???AAUB?sA?}A-
?A???@??BL?A?A ?L@?$A@?'xA???A?{?A??A;?A??A? ?A?{?AЀ?A8?{A??JA??JAk `A???C?w?A@M?AO?C?g?A??A?2???NA??@??iA?{?@???A???A=C??E@?ވ@^??Aށ5B??AԾ?A??q=|eM@?C?@??%A??sAU0A?rMA>?@q?hA??A???A?phA`?/A}?AB??@??jAh??A??TC?_?Aƿ2A??)A?*B??Aۗ=A.?A??W@?B?A?{@?	@?0                     I      	      X      )   h         J      ;   ^                         /               ?            
   	               M               
            *   ?                     5         ;   '                     +            9         /         9      5   M               !                        5                              -                              !                                                                  -                        %   ;                                                                                                                                    M??aN2aN?=NV??N%??O%}?O?ȘN??N;??NC?mP>??NٳO`(XO??]N?=jNR??O??{N??#O?BPqPM??ONe??N?q?N?h O?nO-?N??N???NkixN?<OL?NR?~ON
?N<?N?	N??TNĜ?Nj
N??>NE??Ol?N?¨O??jM?ӷN?O?O??OM?7N?`5NNO?@O?<O?O??oN?E?N QANBAHM??N???N??Om?%N???N4O2O}?O7N  ?  ?  ?  ?    ?  ?  ?  ?  ?  ?  ?    
d    ?  ?  ?  ?  ?  ?  e     V  ?  <  K  	?  ?  n  ?  ?    2  g    ?    ?  M  ?  @  	H  ?  ?  ?  r  ?  ?  ?  5     
4    ?  7  _  B  k  	  v  ?  
?  .??9X??t??#?
?#?
?o?ě?<?o?o???
??o<???;o<T??=T??;?o;??
=?w<T??<ě?<???<o<t?<t?<D??<ě?<u<?t?<???<?9X<?9X<???<???>1'<???=+=+<?`B<???=o=C?=?P=?P=??=?w=0 ?=?w='??=#?
=D??=L??=Y?=T??=?%=Y?=]/=u=?%=?o=?C?=?{=??
=Ƨ?=???>$?????????????????????CNQ[gmig[NCCCCCCCCCCww??????????????wwww-$$)/<?B?<1/--------ONBABDO[a^[OOOOOOOOO????????????????????????????????????????????????????????????????????????????????????????????????????,+/<Haz????????nH<0,),-*)
#/<BDCC<+#
????????? ???????63<<<HUX_WUHA<666666?)0575)???????????
(-/-#
???zyz????????????zzzzz????
#FMPOKD</&	???"BNt????????tm`NBzxvz~????zzzzzzzzzz"+/;<C;/""????????????????????')+16:BGFBA6,)~??????????????????~//9;AHTaba]WUTLH;3//gffhtx}xthgggggggggg?????


?????????????????????????????????????????TMKTaemqz??}zsma_TTTmmoz{?????zxpmmmmmmm	)58BDEC>52)	93:<HUVVUH><99999999|}??????????||||||||???????

?????????????????

?????,,/<?HIH</,,,,,,,,,,???????	???????????????????????????????????????????????????????????????????????????????????????????????????????/,./<HNTUZUNH<7/////????????????????????????????????????????		
#0:<B:0-#
	TZ[ahqskh[TTTTTTTTTT)BUZRKB5)
!')/;HMRT[^ZPH;/"(%&,/3<HNSUSNH</((((?????????? ???????????

????????????????????????????W[gptvttgd\[WWWWWWWW
?????
zww{}?????????|{zzzz????????????????????)5<>;6-)?
 #//430/(#
?????????????????????????????????????????njinoz??????????znnn??????????????r?l?r????????????????????????????????????????????????????????????[?g?p?t?t?t?t?k?g?[?Q?P?N?M?N?T?[?[?[?[?????????????????????????????????????????l?l?t?y???????????y?w?l?l?l?l?l?l?l?l?l???*?6???C?I?E?C?:?6?*?????
?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????Ҿ(?4?8?A?F?A?:?4?(?'??%?(?(?(?(?(?(?(?(??(?A?Y?b?j?h?^?W?N?A?(???????????????????????????????????????????????`?m?y??????????????y?m?`?R?G???H?T?Z?`?????????????????????????????????E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?Eپf?i?i?m?o?f?^?Z?W?U?V?Z?`?d?f?f?f?f?f?f?????׾????????????׾ʾ?????????????????E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E??"?;?H?P?X?Y?U?/?"??	?????????????
??"?????	??#?-?%?????????e?[?W?c?s?????????Y?e?r?z?~???~?r?i?e?Y?X?Y?Y?Y?Y?Y?Y?Y?Y?T?a?a?g?m?o?o?m?a?W?U?T?T?S?T?T?T?T?T?T?????????????????????????|?|????????????ÇÓÙÓÑÓàâàÛÓÇ?|?z?q?z?~ÄÇÇ?????????	????????߻ڻܻ???????"?#?/?2?;?@?C?<?;?:?/?"???	???	??"?A?M?Z?]?e?Z?M?A?;?=?A?A?A?A?A?A?A?A?A?AD?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D??ûлٻܻ????????ܻһлû????ûûûûûû???????????????????????āčĚĞĦĦħĦĤĚčā?|?t?m?s?tāāā????????????????????????????????????????)?4?8?7?6?-?)?????????????????n?zÇÈÎËÇ?z?y?r?n?m?n?n?n?n?n?n?n?n???????ùĹŹù???????????????????????????????????????????????|?s?x?????????????'?4?@?D?M?Y?\?Y?O?M?@?4?'?&?????¤¡???ʾ׾?????????????????׾ʾǾ?????????ÇÓßàäàÔÓÑÇ?ÁÇÇÇÇÇÇÇÇ?M?Z?f?s?u?t?u?s?f?Z?M?A?;?7?A?D?M?M?M?M?!?-?8?:?@?<?:?-?"?!?!????!?!?!?!?!?!?g?????????????????????s?g?Z?S?N?O?R?Z?g?
?	????
??????
?
?
?
?
?
?
?
?
?
?????????????????????????????????"?.?;?G?T?X?U?T?R?G?;?.?"??????"?"???????ݿ?????????????ݿϿ????????????f?s?????????????????????s?k?f?[?Z?d?f?:?F?G?S?X?S?F?:?8?9?:?:?:?:?:?:?:?:?:?:????????????y?m?T?G?=?9?7?<?G?T?`?m?y???
??0?<?I?S?e?l?b?U?R?<?#??
?????????
D?EEEE&E*E6E0E*EEED?D?D?D?D?D?D?D??U?a?n?zÇÍÒÐÍ??n?a?U?H?=?:?9?:?H?U????????????????????????żž???????????ƽ????????????????????????????????????????{Ǉǃ?{?w?o?b?b?_?b?o?t?{?{?{?{?{?{?{?{āĀ?t?q?h?^?h?tāăĂāāāāāāāāā?нݽ????????
????????ݽֽнϽннн?¦²¿????¿²¦£¥¦¦¦¦¦¦¦¦¦¦???????ƺɺкϺɺ??????????~?{?z?}?~????ǭǰǲǭǫǡǠǔǈ?{?z?{?|Ǆǈǒǔǝǡǭ?.?:?=?G?J?G?<?:?0?.?!?!?!?*?.?.?.?.?.?.?ּ????????????????ּʼ??????????????ʼֻ??????????????????????????????????????? m  O / 5  H ? R . , e  # F ? + , 5 h x q , X 7 S ?  S 5 + i  N \ F > L W < + 0 " ` # : w E h A E &  K X L { n 3  T ! B &  (  ?  ?  q  A  Z  h  }  u  _  C  1  ?    ?      ?  ?  ?  N  ?  ?  ?  ;  6  9  ?  ?  3  )  x  ?  k  T  ?  ?  ?    s    ?  x    ?  %  h  &  J  ?  x  A  m    W  f  ]  ?  2  ?  ,  J  !  P  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  D?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  }  y  u  q  m  i  e  a  ]  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  Z  1    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  y  |  ?  ?  }  q  c  T  D  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  t  g  Y  K  <  .          	      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  d  O  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  r  K    ?  ?  ?  J    ?  7  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  ?  ?  ?  3  ?  ?  r  "  ?  ?  ?                  ?  ?  ?  ?  p  M  *    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  r  ]  C  )    ?  ?  ?  ?  `  9     ?   ?  z  ?  >  |  ?  ?  ?  ?  ]    ?  b  ?  ?  '  ?  ?  ?  =   ?  ?  ?  ?  ?  ?  	                      	      ?  g  ?  ?  ?  ?    ?  ?  ?  ?  ?  h    ?  z    ?  M     ?  ?  y  7  ?  	`  	?  
  
F  
^  
c  
X  
2  	?  	[  ?  ?    ?    ?    ?  ?  ?  ?  ?  ?  i  I  )    ?  ?  ?  n  O  .      ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  {  w  r  n  i  ?  ?  	  5  R  g  ?  ?  ?  ?  ?  ?  |  ;  ?  t    ?  Q  ?    %  .  6  ;  ?  =  7  .       ?  ?  ?  t    ?  ?  I  ?  V  n  r  q  ?  ?  ?  ?  ?  y  2  ?  ?  N     ?  ?  H  ?  ?  ?  6  w  ?  ?  ?  p  A    1  4    ?  v    z  ?      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?      3  e  T  B  1       ?  ?  ?  ?  ?  u  ^  G  0       ?   ?   ?     ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  a  >    ?  ?  ?  V  L  A  ;  =  @  ;  0  &        ?  ?  ?  ?  ?  d  G  +  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  a  <  ?      a   ?  <  8  5  1  (        ?  ?  ?  ?  ?  ?  x  n  f  a  ]  X  F  G  I  K  K  J  J  C  8  .  "      ?  ?  ?  ?  ?  ?  ?  	4  	|  	?  	?  	?  	?  	?  	?  	?  	?  	u  	@  	  ?  z    ^  ?  ?  ?  ?  ?  ?  ^  9    ?  ?  ?  ?  ?  |  Z  +  ?  ?  ?  V    ?  n  o  q  s  t  t  m  f  _  X  S  Q  O  L  J  E  @  :  4  /  ?  ?  ?  ?  r  Y  =    ?  ?  ?  ?  \  (  ?  ?  A  ?  ?  '  ?  ?  ?  ?  ?  ?  ?  ?  x  R  )    ?  ?  ?  z  W  1    ?     ?  ?  ?  ?  ?  3  ?    ?  ?      ?  U  ?  ?  
?  ?  ?       #  '  +  /  1  2  2  0  ,  %         ?  ?  ?  ?  ?    ?  ?  ?  ?    e  g  ?  g        ?  ?    ?  ?  M  ?         ?  ?  ?  ?    ?  ?  ?  ?  ?  ?  ?  }  \  >  6  S  ?  ?  ?  ?  ?  ?  ?  }  o  `  P  ;  #    ?  ?  ?  ?  ?  ?              ?  ?  ?  ?  ?  _  <    ?  ?  ?  ~  V  -  ?  ?  ?  |  o  `  R  D  7  1  7  5  -         ?  ?  ?  6  7  F  K  I  B  :  /      ?  ?  ?  ?  ?  ?  h  R  :  ?  e  ?  ?  ?  ?  ?  ?  ]  7    ?  ?  ?  S    ?  c  ?  ?  &  ?    3  =  9  1  #    ?  ?  ?  ?  o  E    ?  ?  ?  K    ?    {  ?  ?  	  	/  	C  	G  	8  	  ?  ?  c  ?  q  ?    0  ?  ?  ?  i  P  6      ?  ?  ?  ?  ?  ?  ?  {  b  H  .    ?  ?  r  ?  ?  ?  ?  ?  ?  x  X  0    ?  ?  I    ?  |  g  ^  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  u  i  ^  Q  D  6  )  q  k  K  (    +    ?  ?  z  1  ?  ?  o  4  ?  ?  ?  Y  ?  ?  ?  ?  ?  ?  x  q  i  a  V  J  >  2  $      ?  ?  k    ?  ?  ?  ?  ~  y  q  h  `  W  K  <  ,      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  w  s  x  v  i  U  8    ?  ?  M  ?  ?  ^  )  -  *             
  ?  ?  ?  q  1  ?  ?  ?  s  ?  ?     
?  
?  
?  
d  
1  	?  	?  	}  	7  ?  ?  -  ?    U  {  ?  ?  ?  
  
  
-  
4  
-  
  	?  	?  	?  	\  	  ?  ]  ?  b  ?  ?    &  q      ?  ?  ?  ?  ?  ?  ?  ?  ?  @  ?  ?  K  ?      ?  +  ?  ?  ?  ?  ?  ?    q  a  P  @  /         ?   ?   ?   ?   ?  7  .  $      ?  ?  ?  x  ;  ?  ?  X    ?  V  ?  ?  D  ?  _  Q  D  6  &      ?  ?  ?  p  <    ?  ?  |  Q  ,    ?  B  @  <  6  %      ?  ?  ?  ?  ?  k  H    ?  ?  ?  J    k  n  r  u  o  h  a  V  J  >  1  $    	  ?  ?  ?  ?  ?  ?  ?  ?  	  	  ?  ?  ?  ?  T    ?  ?  .  ?  N  ?  5  =  ?   ?  v  X  7      ?  ?  ?  ?  ?  N    ?  ?  K    ?  ?  Y  ?  ?  ?  ?  ?  ?  ?  ?  ?  e  B    ?  ?  ?  ?  U  .  	  ?  ?  
F  
?  
?  
?  
?  
z  
Y  
!  	?  	?  	=  ?  ?    ?    b  S  ?   ?  .  
?  
?  
j  
'  	?  	?  	`  	  ?  p    ?  %  ?  ?  8  ?  ?  2