CDF       
      obs    M   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�9XbM�     4  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N
�   max       Pn+u     4  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���w   max       <�h     4      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?c�
=p�   max       @F��Q�       !H   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vU�����       -P   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @Q�           �  9X   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @˼        max       @��@         4  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��x�   max       <���     4  ;(   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B4�d     4  <\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��v   max       B4�I     4  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >A��   max       C��S     4  >�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >6�   max       C��     4  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          O     4  A,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          5     4  B`   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          1     4  C�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N
�   max       P$     4  D�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?� ě��T   max       ?п�[W>�     4  E�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���T   max       <�h     4  G0   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?c�
=p�   max       @F��Q�       Hd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�`    max       @vU�����       Tl   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @Q�           �  `t   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @˼        max       @�e          4  a   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         FV   max         FV     4  bD   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�bM���   max       ?мj~��#       cx            ;      
   &                  O         =      
      	      
               "      )                           D   -   +   
                                                      	            (                           
            %   N���N���N�!�P2^N�FPNrUO��[N�I4N[�O$yRN�YN��IPo�Nt7�N��Pn+uN
�N�fdNi��O'#O�:N��O"��N�0�P6ZO6�gO�ފNL5�O��N��/O���N�g�O]�N���O�FNd�aOX�O���P��P�"N��N�	O�U1N���O��XN�a�N�otN�̏O#�O��NuޣOz�cOI��O4HO�i
N"�N���OO�INmn�O�xGN\.N�oP!�FN��]OP-�O
1'N�8�N�.#OKfWN�N�WO��N���N�,�O:��O���Np2F<�h<���<u<49X;��
;��
;o%   �ě��o�49X�D���e`B�e`B�u��o��C���t����㼛�㼣�
��1��1��1��9X��j��j���ͼ��ͼ�������������/��`B�����o�+�+�+�C��C��C��C��\)�t��t���P��P��w�#�
�''',1�0 Ž0 Ž<j�@��H�9�P�`�]/�]/�]/�]/�e`B�ixսixս�%��o�����\)��hs��hs���w���w���w��������������������#/473/+#
DOQ[ahlnh[OOJHDDDDDDGar������������mTHOG�������������}����������|}}}}}}}}B[v~����������g[NIBBU[hqtv�����tqhg[UUUU)/6BFEB6)%��������������������(/4<HUXaelbaUTH</,((��������������������������
�������GIUYbn{��{nbUIGGGGGGY[gmng\[YPYYYYYYYYYY������������tt�����������������������������������������|���������wx||||||||$*-26CORYbhg\OC65*&$�BKOKPd[OB)��GHQUaabknoonhaUHA?GG����������������������������������������#.<Ibn{����{U<0)"# #GHOTabmvz��zmaTLHFEG��������������������(&���������')������##(/<HUWURLH</+#####(),6<?CIJB6)tt}������������ztptt[hmt���������th[WQQ[5<ILPUVWXWURIC<<:875otw�����������tmmloo����� ��������������������������������������� ����������)BNgp{||xtg[N>)��������������������cgkt}���������tggacc����������������������):N[\OB6)�����	������������������������������HHITX[[XTHGCDDHHHHHHX[fgrsstuvtg][RRXXXX�������������������������������������������
#/+#���������~������������z~~~~~~/5BN[gigfea[NJB<1**/htu|���������������h0<?IUW]\[ULI<20#LN[gt���������tjZZNLjnz����zncjjjjjjjjjj�������������������������������������������������������������
#0DRMLKA<0#
������

���������������������������������%)5B[����������gbB,%��������������������������	������������������������������_bfnxyuttnb]\]______y{���������{zwvwyyyyekt���������xpigbfde��������������������TUajnptx||zyna_USOOTnt~������������ztmnn|�����������zx||||||_anrz�����zna_______��������������������)5GG?4)�  

����� ����!�-�5�-�-�!��������ìäàÝÛàìùý��������ùìììììì�Y�V�Y�_�e�l�r�~�����~�~�r�e�Y�Y�Y�Y�Y�Y������ļĹ���������
��#�0�I�b�_�M�I�0��������������������������������������Ľ����Ľнݽ���߽ݽнĽĽĽĽĽĽĽĿ����y�m�]�c�a�X�`�d�{�������׿ؿοĿ����<�9�5�/�.�/�9�<�H�U�]�Z�U�I�H�>�<�<�<�<�#��#�'�,�/�<�?�D�>�<�/�#�#�#�#�#�#�#�#�лû������������ûܻ�����������ܻоA�9�4�2�0�2�4�=�A�M�Z�Z�Z�Z�U�Q�O�M�A�A����������
������
�����������������f�Y�M�5�4�@�M�f���������ļɼļ�������f�ʼȼ������������ʼͼѼҼϼ̼ʼʼʼʼʼ��z�y�u�z�������������z�z�z�z�z�z�z�z�z�z���j�b�m�r�l�y�������Կ��������忸�����H�B�=�D�H�U�Z�V�U�O�H�H�H�H�H�H�H�H�H�H���������ʼּ��ּʼ��������������������g�]�Z�W�Z�g�s���������s�g�g�g�g�g�g�g�g�`�X�T�G�=�=�D�G�T�`�m�y���~��z�y�q�m�`�M�A�A�6�1�@�M�f�r���������}��������Y�M�Z�X�N�G�A�<�A�N�U�Z�g�i�s�u�x�x�s�g�Z�ZìàÝÛÚÝàäìù��������������úùì����������"�(�1�5�7�5�(������g�N�F�D�C�F�O�]�s���������������������g�������������������������	���������������������������$�0�7�@�=�4�0�$���������������������������������������ʼ��������������ʽ��.�2�0�(���������������������������������������������޾s�Z�M�A�8�A�H�Z�f�s�������Ǿ���������s�����x�t�m�r�x�x�{���������������������������������������Ľ̽׽ݽ�����ݽн����������(�4�A�I�M�R�M�C�A�4�(���N�E�A�;�=�A�G�N�Z�g�i�s�v�y��s�g�Z�N�N�� ��������������� ���������@�3�'����	���#�3�e�~�����~�k�Y�L�@�N�B�?�F�S�g�������������������������g�N�����������������������������������������r�~���������ɺֺ����������ɺ�������ھ׾Ӿ׾������	���	�	������s�n�s���������������������������������s�������|�y�y����������������������������������������� �	�����	��������������׾оʾɾ¾������ʾ׾���	������ܾ��N�L�N�N�Z�g�s���������s�g�Z�N�N�N�N�N�N���������������������������������������ҿ	�����������	���"�)�.�"��	�	�	�	�s�g�U�N�E�M�Z�g�s�z�������������������s�ݿϿϿҿֿܿݿ�������)�+��������6�0�*�(�*�2�6�C�G�O�P�P�O�C�6�6�6�6�6�6�)������)�4�B�N�g�l�k�g�b�[�N�B�5�)���������������������������!�����������������������Ƚнӽ̽Ľ����������$�������$�0�=�V�o�t�y�u�o�V�I�5�$�H�H�E�E�H�T�U�Z�Z�U�H�H�H�H�H�H�H�H�H�H�Ľý��ĽĽнսݽ����ݽнĽĽĽĽĽ����������������������������&���������������� ��������������������]�8�+�'�$�-�:�F�S�_�l�p�v���������x�l�]�ɺ����ɺֺغ���ֺɺɺɺɺɺɺɺɺɺɾ����������������������������������������t�i�m�xĂ�y�yāčĳĿ������������ěā�t�»������������»ûǻлԻܻ���ܻлĻ»��ܻӻӻػ������� ����������U�P�M�N�U�b�n�{�|ŇŔśŗŔŇ�{�n�b�U�U�л˻û��ûлܻ����������ܻлллллм��	���'�.�4�;�@�F�B�@�4�'�����¨²��������������������¿²¦���
��������
��#�/�0�<�?�@�<�0�#���������������ùϹܹ����޹ܹҹϹù�����ÿùøììãìù�����������������������#���!�#�/�<�H�N�S�H�F�<�/�#�#�#�#�#�#�Ŀ¿����������Ŀ̿ѿӿٿ׿ѿĿĿĿĿĿ��6�)�#��������&�)�6�:�I�L�P�O�B�6����ùõîïö����������������������EEEEE*E7ECEGECE@E7E*EEEEEEEE L < ` P O 1 � h Y / a % 9 � U F ^ % C > R 6 K A # = @ ] O N A A F \ % |  _ 4 & + � G 7 G b : _ \ J / 1 9 F 3 x 6 < ` @ N ' ^ -  T d N M X X C - g b F :    �  �  �  B    �  ,  �  �  n  9  �  |  "    
  O  �  c  u  �  �  �  �  �  �  �  �  �  �  �    �    A  �  �  �  �  �  �  @  �  �  c  �  �  �  �  �  t  �  �  �  �  m  �  �  �  �  2  :  �  �  �  D  �  �  �  %  )  5  �  �  �  (  }<���<t�;o�D���o��o�\)��o�D����󶼴9X��/��j��C���t����-��1��`B���ͼ�`B�@����<j��/�D���8Q�ixռ���+���]/�8Q�#�
�0 ŽD�����<j���ͽ��-����49X�'T����㽁%�0 Ž0 ŽD���P�`�y�#�<j�m�h�m�h�]/����L�ͽL�ͽixսaG�����e`B�q����vɽ�����
�y�#�u�}󶽝�-���-��Q콣�
��j���w�ȴ9��xսě�B+ȏB�B�B 
�B��B)z�B	PTB�.B�B!}BD�B|?B"��B'�GB��B+x+B!-�B��BB0�CBmKB�B��B-qB'?9A���B`JBb�B-øB��B8�B�BW�B&��B
d]B1{B"_,B�B"�B�B	��BM�B,B��B�OA��B	9�BR�B�wB�B
��B+<B
��B&E�B	�B=�B"DBk�B�tB%)�B#�B4�dB	a�B�BɲBUB(0�B)�B
?�BBEBQ�B
��B
��B�B`�BnB�nB+��B�B?kB =�B��B)�B	=�BG�BW(B!�BCB~�B"��B'��B�lB+��B ��B�3BԮB0ƁBH�B�Bn'B>�B'��A�o�B��B�sB-��B+{B��B:�BёB&��B
F�B>�B"QnB�BXpB�B	�+B�BE�B��B�NA��vB	<B?�B��B@�B
�@B,�B&�B&�]B	��B�2B!ĬB� B��B$�*B#�PB4�IB	��B��B�0B!�B(?6B)bB
@ B?�B@B
��B
�`B� B@�B��B�	@e��A�6L?�"lA���A�m]A)�AtU�A�R�A�vO@�z\A;�jA�m�@�r@��A�l>At�,AĻ�@�GZA�H)Ai{ @�/GA�(A���A���A���B�*B��A1XArOA�AF;c@��qA(6�A7��A���A���?�:EA���A�w@2��AWԤA�RA�i`A��CAU[A��lA��2A[��A���A�`�B ��A�=�A�JA#sOB
��A�ݟA*ZA��2B��@��@9s�AI�vA�R@��@@��6A��s@���@��ZA��A���>A��Aΐ�A���Ax�A��lA��C��S@e�A��?�8�A�zPA�sA)��At�,A�y�A�@��A;MA���@�]@��oA���Au�A�i@���A��uAh��@���A�w0ÅIA�H�A�s�B��B�qA1P9A��A�v�AE��@�?A)D�A8��A��A��?�'WA���A���@3��AX�<A��&A�~bA�t�AT�/A�A���A\ޕA�=A�CB ��A���A��A"��B8�AŁwA*�tA��gB��@�	0@5 �AJN�Aߎ�@��3@���A���@�H@���A��6A�>6�A�l�A�y}Az�iA���Aφ?C��            <      
   &                  O         >            
      
               #      *                           D   -   ,                                                         	            (                                       %               3         /                  +         5               )            '            +      #                  #   '   )   '         #                                    !                     
   1                                                                                          '               )            '            #      !                     #   %   !                                             !                     
   1                                          N���NT��N�!�O�4N,��NrUO:�3N�I4N[�No�N�@�N��IO*��Nt7�N��P$N
�N�fdNi��O'#O�:N��SN��N�0�P6ZO6�gO���NL5�O�N��/O���Nǅ>O]�N��<N��ZNd�aN��OǛ�P�O�΀Nɒ�N�	O��ON���Ofq-N�a�N�otN�̏O#�O��ENuޣO^��OI��O4HO�i
N"�N���OO�INmn�O�B;N\.N�oP!�FN��O��O
1'N�8�N�.#O��N�N��O��N�f
N�,�O)(O^VANT��  �  �    �  �  ;    �  M    �  C  �  �  X  *  "  b  �  @      !  P  u    �  �  �  Q  I    0  S  �  7  �  
9  �  -  r  E  q  �     �  N  4  �  u  �  �  b  N  �  �  �  }  b  �    �  �  �  d  �  ;    u  D  �  �  �  �    �  �<�h<�9X<u�o��o;��
�49X%   �ě���t��e`B�D���P�`�e`B�u��󶼋C���t����㼛�㼣�
��9X��/��1��9X��j��/���ͽo������h��/��/�o�C�����P�,1��P�,1�\)�C��\)�C����t��t���P��P�'#�
�,1�'',1�0 Ž0 Ž<j�@��L�ͽP�`�]/�]/�e`B�q���e`B�ixսixս�+��o��C���\)�����hs�������T������������������������#////(#DOQ[ahlnh[OOJHDDDDDD[bmrz���������zma]X[��������������������}����������|}}}}}}}}KNT[gt�����{tg[TNNKKU[hqtv�����tqhg[UUUU)/6BFEB6)%��������������������3<>HRU`aUH=<33333333����������������������������������������GIUYbn{��{nbUIGGGGGGY[gmng\[YPYYYYYYYYYY������� ��������~~�����������������������������������������|���������wx||||||||$*-26CORYbhg\OC65*&$�BKOKPd[OB)��HHSUahmmdaUHC@HHHHHH����������������������������������������#.<Ibn{����{U<0)"# #GHOTabmvz��zmaTLHFEG��������������������(&�����#!��������##(/<HUWURLH</+#####&)6;?CFGB=6)qtu~����������|tqqqq[hmt���������th[WQQ[<<DIJQUUVUUIIH?<<;<<qtz�����������tspoqq����� ���������������������������������������������������)BN[goxzzvng[NA)��������������������dgmt������thgbdddddd����������������������)8OYZOB6) ����	������������������������������HHITX[[XTHGCDDHHHHHHX[fgrsstuvtg][RRXXXX����������������������������������������������
"#��������~������������z~~~~~~15BN[dgfeda[[NB=2+,1htu|���������������h0<?IUW]\[ULI<20#LN[gt���������tjZZNLjnz����zncjjjjjjjjjj��������������������������������������������������������������
#0AIOJH?0,#
�����

���������������������������������%)5B[����������gbB,%�������������������������������������������������������_bfnxyuttnb]\]______y{���������{zwvwyyyyst����������{ttnilss��������������������PUWagnoruuncaUTPPPPPnt~������������ztmnn}�����������{y}}}}}}_anrz�����zna_______��������������������	)57DDB=2)

 ���� ����!�-�5�-�-�!��������ìçãììïù��������ùìììììììì�Y�V�Y�_�e�l�r�~�����~�~�r�e�Y�Y�Y�Y�Y�Y�����������������������
���#�#� ��
���������������������������������������Ľ����Ľнݽ���߽ݽнĽĽĽĽĽĽĽĿ��������������������������̿̿Ŀ��������<�9�5�/�.�/�9�<�H�U�]�Z�U�I�H�>�<�<�<�<�#��#�'�,�/�<�?�D�>�<�/�#�#�#�#�#�#�#�#�ܻܻлŻû��ûлٻܻ޻����ܻܻܻܻܾA�7�4�2�4�5�A�M�V�R�M�K�A�A�A�A�A�A�A�A����������
������
�����������������Y�S�M�L�Q�Y�f�r�����������������r�f�Y�ʼȼ������������ʼͼѼҼϼ̼ʼʼʼʼʼ��z�y�u�z�������������z�z�z�z�z�z�z�z�z�z�����r�m�m�}�������Ŀ׿����ݿѿ������H�B�=�D�H�U�Z�V�U�O�H�H�H�H�H�H�H�H�H�H���������ʼּ��ּʼ��������������������g�]�Z�W�Z�g�s���������s�g�g�g�g�g�g�g�g�`�X�T�G�=�=�D�G�T�`�m�y���~��z�y�q�m�`�M�A�A�6�1�@�M�f�r���������}��������Y�M�Z�Z�N�J�E�N�Z�g�s�t�w�w�s�g�Z�Z�Z�Z�Z�ZàßÝÜßàêìòù����������þùìàà����������"�(�1�5�7�5�(������g�N�F�D�C�F�O�]�s���������������������g�������������������������	�������������������������������$�0�4�=�:�0�$����������������������������������������������ɼ���!�*�+�'�������㼽���������������������������������������޾�s�Z�S�M�H�Q�f�s�����������þ���������������x�u�m�s�x�����������������������������������������Ľ̽׽ݽ�����ݽн��������(�(�4�=�A�M�N�M�A�9�4�(���N�I�A�>�@�A�N�O�Z�a�g�q�s�t�u�s�g�Z�N�N�� ��������������� ���������@�?�5�9�@�L�Y�e�o�e�e�Y�P�L�@�@�@�@�@�@�g�Z�N�F�B�J�X�g�s�������������������s�g�����������������������	������������������������������ֺ��������ֺɺ�������ܾ׾����	���	�����������s�n�s���������������������������������s�������}�z�z����������������������������������������� �	�����	��������������ʾž¾����ʾ׾������	���	�����׾��N�L�N�N�Z�g�s���������s�g�Z�N�N�N�N�N�N���������������������������������������ҿ	�����������	���"�)�.�"��	�	�	�	�s�g�U�N�E�M�Z�g�s�z�������������������s���ݿѿпҿԿؿݿ������� �������6�0�*�(�*�2�6�C�G�O�P�P�O�C�6�6�6�6�6�6�)�"�����(�5�B�N�[�g�h�i�a�[�N�B�5�)���������������������������!�����������������������Ƚнӽ̽Ľ����������$�������$�0�=�V�o�t�y�u�o�V�I�5�$�H�H�E�E�H�T�U�Z�Z�U�H�H�H�H�H�H�H�H�H�H�Ľý��ĽĽнսݽ����ݽнĽĽĽĽĽ����������������������������&���������������� ��������������������l�_�<�/�*�&�-�3�:�F�_�l�t�x�������x�l�ɺ����ɺֺغ���ֺɺɺɺɺɺɺɺɺɺɾ����������������������������������������t�i�m�xĂ�y�yāčĳĿ������������ěā�t���������ûлܻݻݻܻлû�����������������޻ܻڻ޻�����	�������������U�P�M�N�U�b�n�{�|ŇŔśŗŔŇ�{�n�b�U�U�л˻û��ûлܻ����������ܻлллллм��	���'�.�4�;�@�F�B�@�4�'�����¦¥¢¦¬²¿������������������¿²¦¦���
��������
��#�/�0�<�?�@�<�0�#�����������������ùϹܹݹܹܹйϹù���������ÿùøììãìù�����������������������#���#�$�/�<�H�K�R�H�E�<�/�#�#�#�#�#�#�Ŀ¿����������Ŀ̿ѿӿٿ׿ѿĿĿĿĿĿ��6�)������� �)�6�9�A�B�I�K�P�O�B�6��������ù÷ðñøù��������������������EEEEE*E7ECEGECE@E7E*EEEEEEEE L & ` 7 [ 1 2 h Y 0 X % & � U 6 ^ % C > R ? F A # = B ] @ N < ; F M ! | B [ - & & � G 7   b : _ \ < / ) 9 F 3 x 6 < ` A N ' ^ -  T d N @ X : C , g ] 0 :    �  c  �  e  d  �  �  �  �  �  �  �  m  "    �  O  �  c  u  �  �    �  �  �  a  �  �  �  n  �  �  �  �  �  �    z  �  �  @  �  �  �  �  �  �  �    t  �  �  �  �  m  �  �  �  q  2  :  �  �  A  D  �  �  @  %  �  5  �  �  �  �  n  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  FV  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  o  b  T  F  8  ^  y  �  �  �  �  �  �  �  �  �  �  u  W  5    �  �  �  �    �  �  �  �  o  [  I  7    �  �  �  G  �  �  f    �  p    �    w  �  �  �  �  �  v  L    �  �  `     �  �  2  �  8  I  Y  h  w  �  �  �  �  �  �  �  �  �  z  G  �  �  �  v  ;  .  "    	  �  �  �  �  �  �  �  �  �  �  }  i  T  >  (  �  :  <  B  P  �       �  �  �  �  [    �  _  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  e  P  ;  $     �   �   �  M  M  M  M  G  A  ;  ,      �  �  �  �  �  q  J  �  �  :  �  �  �  �  �  �  �  �      �  �  �  �  �  d  2  �  �  $  s  t  u  z  ~  �  �  |  o  _  J  4      �  �  �  �  �  �  C  &    �  �  �  �  a  <    �  �  �  F    �  Y  �  �  4  4  �  4  �  �    G  l  �  �  f  :    �  t  �    �  7   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    {  v  r  m  X  W  V  U  T  R  Q  O  K  H  E  A  >  >  F  N  V  ^  f  n  �  �  �      )      �  �  t  )  �  y    �    �  �  ?  "      �  �  �  �  �  �  �  }  Q  &  �  �  �  �  i  G  %  b  a  a  `  `  _  ^  [  V  M  E  >  1  !    �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  t  e  R  9  !    �  �  `  @  -      �  �  �  �  �  �  s  ]  G  4  (     �   �   �   s        �  �  �  �  �  �  �  ~  p  c  H  "  �  �  z  [  7  �  �    �  �  �  �  �  �  �  z  `  C  %    �  �  �  j  >  �  �  �             �  �  �  :  �  �    �  5  �  7    P  J  E  ?  :  5  0  *  $        �  �  �  �  �  c  *   �  u  e  U  H  ?  3  '      �  �  �  �  c  <    �  �  U   �      �  �  �  �  �  �  o  6  �  �  t    �  P  �  *  �   �  �  �  �  �  �  �  �  �  b  4  �  �  t    �  K  �  ~  0  �  �  �  �  �  �  �  �  �  �  z  l  _  P  @  0     �  �  �  I  Q  �  �  �  �  �  �  ~  [  3    �  �  <  �  x  �  u  �  *  Q  G  <  2  '      �  �  �  �  �  �  �  �  {  h  W  E  3    E  I  B  /    �  �  �  k  6    �  �  j    �  �    8  �      �  �  �  �  �  �  �  �  n  M  )    �  .  �  �  )  0  !       �  �  �  �  �  �  �  �  j  N  .      �  �  �  ,  <  I  P  R  S  F  )     �  �  f  )  �  �  b    �  ^   �  �  �  �  �  �  �  �  �  z  ]  0    �  �  l  8  �  �    �  7  "    �  �  �  �  �      �  �  s  a  L  1    �  �  �  c  Q  :  "    *  c  �  �  ~  r  d  W  I  8  "    �  �    	�  	�  
1  
7  
&  
  	�  	�  
  	�  	�  	�  	T  �  t  �  1  ^  o  �  �  �  �  �  �    X  )  �  �  p  +  �  �  l  9     �    �  �  )  ,  ,  -  $    �  �  y  9  �  �  I  �  s  �  q  �   �  p  q  q  d  W  I  :  *      �  �  �  �  \  0    �  �  �  E  >  8  2  .  )  '  %  #      �  �  �  �    V  "   �   �  g  p  k  ]  J  7  $    �  �  �  �  �  ]  3  	  �  �  m   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  n  a  S  E  7  �  �  �  �  �  �  �  �  �  s  C    �  �  �  t  !  �  �   �  �  �  �  �  �  w  g  V  F  3  !    �  �  �  �  u  T  2    N  G  A  ;  6  2  (        �  �  �  �  �  �  �  �  n  L  4       �  �  �  �  �  �  p  Q  0    �  �  }  :  �  �  r  �  �  �  �  }  n  b  Y  N  A  2  #    �  �  J  �  �  -   �  ]  q  r  b  M  4      �  �  �  �  ]  .  �  �  �  B    �  �  �  �  �  �  �  �  �  s  ]  G  1      �  �  �  �  �  �  �  �  �  �  {  t  k  _  Q  @  -      �  �  �  �  S    �  b  _  _  \  T  H  8  #    �  �  �  �  X    �  �  2  �  R  N  =  -      �  �  �  �  �  �  �  }  f  B    �  |  "   �  �  �  �  �  �  �  _  6    �  �  �  b  1  �  �  h  	  �  L  �  �  �  �  �  }  y  v  t  s  r  q  h  `  U  G  9  +      �  �  �  �  �  �  {  j  Y  E  0    �  �  �  �  �  ~  y  s  }  p  `  K  5      �  �  �  �  �  ~  f  O  ;  #      <  b  a  `  a  d  p  �  �  �  �  �  �  �  �  �  �  �  �  f  E  �  �  �    ~  y  m  a  h  d  W  H  8  2    �  c    �  s      �  �  �  �  �  �  �  �  �  �  x  q  i  a  o  �  �  �  �  �  s  ]  J  @  5  *        �  �  �  �  �  �  z  c  K  �  �  w  e  T  U  2    �  �  �  �  �  �  \  "  �       �  u  n  n    �  ~  r  c  S  C  2  !    �  �  �  �  �  Z  #  '  ?  S  `  d  c  \  M  7    �  �  �  g    �  j    �    �  �  �  �  �  u  g  X  J  =  0  #      �  �  �  �  �  �  ;  1  &         �   �   �   �   �   �   �   �   �   �      p   b   T             �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  W  `  j  s  u  o  g  [  J  5    �  �  �  \    �  E   �   P  D  ;  2  )  #  &  ,  1  '       �  �  �  _  (  �  �  w  8  �  �  �  �  �  �  O    �  �  Z    �  E  �  E  �  5  �  �  �  i  K  ,    �  �  �  �  �  f  B        �  �  �  �    �  s  �  �  �  w  g  Q  .    �  �  i  .  �  �  $  �  X  �  ~  �  �  �  �  �  �  �  m  Y  ;    �  �  �  �  �  �    �            �  �  �  �  [  +  �  �  _    �  �  t  ?    �  �  �  �  �  �  �  �  c  :    �  �  ]    �  3  �  /  W  �  �  �  �  �  �  �  �  �  �  }  p  a  Y  _  [  H    �    �