CDF       
      obs    K   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��t�k     ,  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       NC(   max       P��     ,  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �}�   max       <���     ,      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?@        max       @FǮz�H     �  !0   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @vk��Q�     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P�           �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @���         ,  98   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       <D��     ,  :d   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�G   max       B4��     ,  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B4�x     ,  <�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�(   max       C�:�     ,  =�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >:��   max       C�8�     ,  ?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          V     ,  @@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;     ,  Al   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;     ,  B�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       NC(   max       P��     ,  C�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��*0U2b   max       ?��p:�~�     ,  D�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <���     ,  F   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?@        max       @F������     �  GH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @vk33334     �  S    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @P�           �  ^�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @���         ,  _P   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F8   max         F8     ,  `|   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?p�)^�	   max       ?��p:�~�     p  a�                  8                  
   	         
         5         V                  !            &            2      6         2                     
         $         
                     1            0   )   +         	      
      $   N�s/N[T1Ng�N�O4cQO�~O��OL�QNk+OM0Oi�N��N�$NkN��^N�9
N��Og�P��O�R<O���P��O&%>Nc��N��MO($�N�d�O�@\O3�'P6�N�ĦO�W�O%J�OV7N��rO�n�O
�7O΋1O��O��O��O��fND�sP�NjG�N�0N�]�N�ctO4��N���O���N�QOrسN�OD�&N�I�N�,�N3�8O�@�O��eO�A�N��NC(N饈O���O��+O�ɀO$bO�aN���Nz/�O�O��O�&�O{�U<���<D��<#�
<t�;�`B;�`B;D���o�o�D����o�ě���`B�t��49X�49X�49X�D���T���e`B�e`B�u��o��o��C����㼛�㼣�
��1��1��j�ě����ͼ��ͼ��ͼ�/��`B��`B��`B��h�������o�o�o�+�+�\)�\)�\)�t���P����w�#�
�0 Ž0 Ž0 Ž49X�49X�8Q�H�9�P�`�P�`�Y��e`B�e`B�e`B�m�h�m�h�u�u�y�#�}��������������������������������~�������������	���������������� &�������)*7?ECDC65)'q�����������������tq"#)/<HU_iaYRH<#!"��"$),*+,1/#
���������������������#-/<OU]a]UH</#56BPOK^c`[OBA>;63665(/1<HUZUSMHC<5/-((((
"#*,#
�#$&#��������������������)6<BFLOOPOHB62);<IUaVUIB<;;;;;;;;;;^cmz����������zumh`^#<Zz��������{U<�������������|zvuux�X\hku�������������uX�������������gmrz{�����������zmg�����	����������������������������kmz������������ztomk_anz����|zqna[ZZ____�)5BGOSSOGB5)"��X[]ht�������{thd[YUX����������������RTWahmtunmidaTSRRRRRHan����������zaUH>@H:;HRVXZaaTRMH;8558<:���������������������������������������������������������������t���}����������xnnrt*<AGIOUWaefb`ZI<0&%*��������������������w���������������zsnw����������������������������������������uz����� %��������zu��������������������26BBOP[\[XQOB;6,2222KNT[gnpnkhg][TNMKIKK�  �������INW[gtu������}tf[SLI��������������������#0<CFFKJGA5#��
 
����������)6<:?>:6)#  )6BO[[STPB62)!y{����������������{yR[_hlt�{tih[XPRRRRRR����������������������������������������25Nafpt�����u[NB9512	#5@A<4,)(��������� �����������TU[acdbbaUTRTTTTTTTT����������������������������������������������������������������������������������

 �������#*0<IXbgcYUI<70(#������	�������������������������������������������������� 
���� #$!�������)4985)����Y[bglt���������g[VVY��������������$�#��������������������������������������������¡¦­««­¦�������������������ĿǿʿͿǿĿ�������������������'�4�@�M�M�D�@�4�1�'���������r�f�V�R�W�f�r������������Լռ������A�>�4�/�*�,�3�A�Z�d�s���������s�f�Z�A�t�h�[�N�B�4�)���)�5�B�N�g�t�t�	���	��"�"�$�"��	�	�	�	�	�	�	�	�	�	¿½¶¿������������������������������¿�T�R�I�T�V�`�m�y�������������y�r�m�b�]�T�B�7�6�3�1�-�6�;�B�O�T�[�e�[�Q�O�B�B�B�B����������������������������������������ìëìöù��������ùìììììììììì�U�a�n�r�x�n�m�a�U�K�I�T�U�U�U�U�U�U�U�U�"������"�,�/�;�<�;�:�3�/�)�"�"�"�"�ʼȼ����ʼּ׼߼ּͼʼʼʼʼʼʼʼʼʼ�ìàÓËÒÓÚìù������������������ùì�����j�O�5��"�N�e�����������������������L�F�I�Q�Z�f�s���������������������f�L�ʾʾ����������������������ʾ׾����ʼ���r�l�j�r����������ּ����ּ��������ݿԿѿĿ����ȿѿݿ����������������;�4�;�C�;�1�;�F�H�J�T�V�X�T�O�H�;�;�;�;�s�i�f�Z�X�M�I�M�Z�f�s�������������s�s��������������$�'�0�<�7�5�/�$������m�j�f�b�m�q�z�}���������������z�m�m�m�mŠŗŏőŚŠŭ��������������������ŹŭŠ���������������������
�������
���񿫿��t�m�p�l�s�u�������Ͽѿֿ�����ѿ���ƵƳƫƧƣƧƳ�������������������������+�����(�/�<�H�U�[�[�b�e�f�a�X�U�<�+�����|�������������������������������������׾ʾ����ʾҾ���� �	��!�!��	���𾌾���v�z������������������������������N�A�5�.�2�>�N�Z�g�s���������������s�g�NàÞÙÖÔÖàáìùú����������ùìàà�˻û��������������ûܼ��+�)����ܻ˾�����ݽսнǽнݽ���(�4�H�L�B�(���c�Y�L�K�G�L�Y�^�e�r�~�������������~�r�c������ĿĴĭĬı�����������������
�
�����3�2�B�P�Z�g�s�����������������s�g�W�A�3ŹųŭŢŪŭŹſ������żŹŹŹŹŹŹŹŹ�Z�V�V�[�g�������������������������s�g�Z��ٺֺѺֺ������������������������������������������������������������������������������Ŀſѿؿѿ˿Ŀ��������-�'�'�-�5�:�F�Q�S�X�S�G�F�:�-�-�-�-�-�-�����������	��"�$�"����	����E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���̻����ǻлܻ������'�@�L�@�����D�D�D�D�D�D�D�D�EE	E
ED�D�D�D�D�D�D�D��Z�U�D�A�;�8�5�A�M�Z�s����������y�s�f�Z�����|�����������������������������������4�'���'�4�H�M�f�r�����w�h�\�Y�M�@�4�ù����������ùϹйܹ߹�ܹϹùùùùù�ŠŠŠŤŭŹ������������ŹŭŠŠŠŠŠŠ����������*�1�*�������������������������)�0�1�5�B�N�[�g�m�g�\�B�5�������������������������	�	�����������������z�r�u������������ûͻλ˻ƻŻû����������(�5�;�6�5�(�����������������������������������������������ѿG�>�;�4�1�.�'�.�;�G�T�V�`�`�c�`�]�T�G�G���׾Ⱦ¾¾ʾ׾����	��!�#�#�"�����@�4�@�J�P�_�r�������������������r�e�Y�@������ٺٺ���!�-�4�F�S�c�e�_�X�:�!��!������!�)�-�:�F�J�R�N�F�@�:�0�-�!�ּ˼ȼʼҼּ����������
��������I�E�=�8�;�=�?�I�V�b�m�m�b�V�I�I�I�I�I�I�������$�0�1�4�0�'�$��������S�N�G�@�:�2�:�>�:�G�S�V�`�l�y�����l�a�S�������Ľнݽ�����������ݽнĽ��������B�9�)�#�-�6�B�O�`�tāčđėĐă�o�h�[�B�0�,�#��������
��-�0�<�I�N�U�Q�I�<�0 Z V ` e N 6  s F O R J L 8 8 f Z ; < 8 _ * 2 T L f @ 2 & b [ M  Z + ( D Z W 9 k V @ � , 5 H 8 M E D } $ � f > F � E +  z � 9 ) T B J \ T J @ l e D    -  �  �  :  �  �  A    +  �  6  �  �  :  �  �  ^  �  _  w  �  �  m  �  �  �  �  �  �  e  �  �    �  �  l  C    X  7  u  9  l    n  �  )  �  �  �  �  `  �  �  �  �  �  �  �    E  \  [    L  �  �  �  K  �  �  J  �  �  �<D��;ě�;��
�D���49X�D���ě���1��o�����49X��o�u�T����j���
��C��+����C��+���ͼ����㼼j���ě��T���\)�8Q�����y�#���<j��㽛��L�ͽ�e`B�#�
���
�H�9�\)�u�t��#�
�m�h�,1�L�ͽixս�hs�0 Žq���D���L�ͽY��L�ͽ8Q콇+��+��j�@��T����o�ȴ9��vɽ������w��o��7L��%��O߽�t��Ƨ𽝲-B�!BD�Bp�B	�BX�B�B�BI�B��B��B��B�4B$�#B[�B!�	B�XB&��A���B'HBB��B4��B�!B ��BiXB"k�B ��B�B��B9�B,�A��B+�A�GB�B:+By�B�BN%B&��B"�TB/gB�B�B�B��Bb:B�BB,�B	s�BrdB%DB��B��B�rB* 'B��BY�B4QB�+BD�B(LB=Bj�BB*-B�B"��B&��B-��B�OB��B��BI�B�tB	�B��B\7BG�B@�B@LB¯B�B��B��B@�B\B�[B$�%B�nB!�B?�B&�<A���B'��B��B4�xBB9B ��B��B"tnB �rB��B:xBC�B, �A�U�B}HA��B��B>�BO�B�pBB1B&�kB"��B��B?XB��B:'B��B�,B�B?wB	C�BI�B%?�B��B��B=-B)�_B�?B��B@lB@hB�wB?B�\BD6B3B>gB?�B"��B&��B-�`B�B�wBAfB9�B��B	B7A1_�A��A�FwAv��@�6>@�E�A>u|A���A�r�A�d�Ak�7A��|@�Q�A�݀A�FTA�(@�.?A͌�A���ADm�ANه@��&A}j A�30AA�B	N[A�גA��A�3�Av�nB�pA�x�A���AWm�AGA&A�A̭f@���A3:Y?�c7A�/�A���A�NA���@H�BA�"�AuR@~,�AY�C�:�@���C�GA@��@�[�@���>�(A��A�s�A��TA��k@��6A�`A�ڽAeIAW)�@��@lcn@sy�A 0B��B	m�A�QA(5�A�9�A��A0�mA�|�A�bfAw@�OC@�b�A=�>A��bA��A�}�Ak	A��@���A�j�AƊkA�s@� �A��A�ОAB>xAN�$@���A}��A�V6ACTB	L�A���A�vPA�~2Ar'�B�AA�o�A��AX.�AF�YA��IA�j@���A6�?��A〧A���A���A�_�@GE�A���At��@{�&AY%�C�8�@�'C�H�AA�@�@��>:��A�e�A��+A��2A�q�@���A�qA��AeAVo@8V@d.B@u,�A�CB�B	�A!�A)%A۪�A�|W                  8                     	         
         5         V                  !            &            2      7         3               	      
         %                              1            0   )   ,      	   
      
      %                     '                                       ;      #   '                        /                        '         '         /                     #                                             !   #                     !                                                            ;         #                        /                        #                                       !                                                                        N�s/N)<�Ng�N�Q�OFON��4OS�O<%BNk+O��N�ĔN�,�N�$NkNz'EN�9
N��Og�P��OEl�N� (O�Y�O�ENc��N��MO($�N�d�O�_O.�P6�N�ĦNq��O%J�O&rN2zO��N�X�O� �Ox�9O��O���O��fND�sO*��NjG�N�0N�N�ctO �,N�t�O��N�QO&/Nl��OD�&N�.4N�,�N3�8O��O��eO0��N��NC(N��OL�O`�PO��POԾO�aNy��NC<gO�N�E�O�\:O{�U  �  �  y  7  N  �  �  N  �  �  �  |    w  F  \  �    M  X  j  
B    �  �  *  �    
  �  !  �  �  �  �  �  �  $  �  R  	�  �  \  �  @  �  &    �  �    '  b  �  �  0  �  �  {  c  �  �  k    N    
  x  %  �  c  %  �  �  �<���<49X<#�
;�`B;��
��`B�D���D���o�o���
��`B��`B�t��D���49X�49X�D���T�����
���ͼ�/��t���o��C����㼛�㼴9X��j��1��j�H�9���ͼ�h���\)��h������h��P�����49X�o�o�\)�+�t�����P�t��,1�#�
��w�'0 Ž0 Ž8Q�49X�m�h�8Q�H�9�Y��ixս�%�u�m�h�e`B�q���q���u�y�#����}����������������������������������������������	���������������	����������#)46=CAB?6)!��������������������"$'/<HNU[^_VOH</'##"�� #+)+*+0.#
����������������������#(/<EHUVWUH</#6BJIO[`^[XOB@<646:66*/5<HTRKHB<6//******
"#*,#
�#$&#��������������������)6<BFLOOPOHB62);<IUaVUIB<;;;;;;;;;;^cmz����������zumh`^#<Zz��������{U<��������������}|||������������������������� ��������yz�������������zsyy�����	����������������������������kmz������������ztomk_anz����|zqna[ZZ____�)5DLNQQNB5)$��Z[`ht��������th\[WZZ����������������RTWahmtunmidaTSRRRRRenyz�����zndeeeeeeee:;HRVXZaaTRMH;8558<:�����	��������	��������������������������������������������������t���������������yppt-<DIU]bcdb^YI<10+('-������������������������������������������������������������������������������������������������������������������������26BBOP[\[XQOB;6,2222LNU[glomkgf[[[NNLJLL�  �������KN[gt~������tig[UMK��������������������#0<@DDGE?3#
��
 
����������),36:8864)'),6BMOPOLB76)''''''y{����������������{yS[dhjt~tthe[YQSSSSSS����������������������������������������45BN^clr��}tg[NB:734	#5@A<4,)(����������������������TU[acdbbaUTRTTTTTTTT����������������������������������������������������������������������������������� 
	����������"#/0<IVbebaUI<90*$#"������	�������������������������������������������������� 
����""
������)275)�����Y[bglt���������g[VVY��������������$�#��������������������������������������������¡¦­««­¦���������������ĿǿʿĿ¿���������������������������'�4�@�J�A�4�.�'�����������|�~������������������������������Z�M�A�7�/�1�4�9�A�M�Z�f�s�}�|�{�s�g�f�Z�t�j�[�N�B�7�)� �5�B�N�[�g�t�z�t�	���	��"�"�$�"��	�	�	�	�	�	�	�	�	�	¿»¿������������������������������¿¿�T�N�T�`�d�m�y�������������y�p�m�f�`�T�T�B�:�6�4�3�6�B�O�R�[�d�[�P�O�B�B�B�B�B�B����������������������������������������ìëìöù��������ùìììììììììì�U�M�K�U�V�a�n�q�w�n�j�a�U�U�U�U�U�U�U�U�"������"�,�/�;�<�;�:�3�/�)�"�"�"�"�ʼȼ����ʼּ׼߼ּͼʼʼʼʼʼʼʼʼʼ�ìàÓËÒÓÚìù������������������ùì�����j�O�5��"�N�e�����������������������f�b�Z�Q�P�Z�f�s��������������������s�f���������������������ʾʾооʾ�������������r�o�n�o�u����������Ҽڼ޼ݼּʼ����ݿݿѿĿĿÿĿ˿ѿݿ�������������ݿ��;�4�;�C�;�1�;�F�H�J�T�V�X�T�O�H�;�;�;�;�s�i�f�Z�X�M�I�M�Z�f�s�������������s�s��������������$�'�0�<�7�5�/�$������m�j�f�b�m�q�z�}���������������z�m�m�m�mŠŘŔőŔŠŭŹ������������������ŹŭŠ����������������������
����
������񿫿��t�m�p�l�s�u�������Ͽѿֿ�����ѿ���ƵƳƫƧƣƧƳ�������������������������H�@�<�;�6�<�H�R�U�X�X�U�H�H�H�H�H�H�H�H�����|���������������������������������������׾˾ľʾ׾ݾ����	������	��������|��������������������������������Z�N�A�:�3�7�D�N�Z�g�s�������������s�g�ZàßÙÖÖàìùù����������ùìàààà�ϻȻû����������ûܻ���)�(�!����Ͼ�����ܽҽݽ����(�4�F�J�A�@�4�(���c�Y�L�K�G�L�Y�^�e�r�~�������������~�r�cĿĹĳİĮĵ��������������������������Ŀ�3�2�B�P�Z�g�s�����������������s�g�W�A�3ŹųŭŢŪŭŹſ������żŹŹŹŹŹŹŹŹ�s�r�k�m�s�v���������������������������s��ٺֺѺֺ������������������������������������������������������������������������������ÿĿѿѿѿƿĿ��������-�'�'�-�5�:�F�Q�S�X�S�G�F�:�-�-�-�-�-�-����������	��"�#�"�����	����E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���λ��û˻лܻ�����'�<�H�@�3�����D�D�D�D�D�D�D�D�EE	E
ED�D�D�D�D�D�D�D��f�a�Z�L�F�@�A�M�Q�Z�f�s�z��������u�s�f�����������������������������������������4�'���'�4�H�M�f�r�����w�h�\�Y�M�@�4�ù����������ùϹϹ۹ܹ߹ܹϹùùùùù�ŠŠŠŤŭŹ������������ŹŭŠŠŠŠŠŠ����������*�1�*��������������������������)�5�B�N�[�e�i�k�d�Y�B�5�������������������������	�	�������������������������������������»ûƻ»��������������(�5�;�6�5�(�����������������������������������������������ѿG�A�;�6�4�3�;�G�R�T�^�`�a�`�Z�T�G�G�G�G��׾˾žƾʾԾ׾����	�����	�����Y�V�R�Y�e�h�r�������������������~�r�e�Y����޺޺����!�-�F�S�`�b�_�S�F�:�!����!� �����!�*�-�7�:�G�P�K�F�>�:�-�%�!�ּ˼ȼʼҼּ����������
��������I�G�=�9�=�I�V�b�k�k�b�V�I�I�I�I�I�I�I�I���	����$�.�0�1�0�$���������S�N�G�@�:�2�:�>�:�G�S�V�`�l�y�����l�a�S���������Ľнݽ�����������ݽнĽ������B�6�,�6�;�B�O�R�c�tāčĔĎĀ�m�h�[�O�B�0�,�#��������
��-�0�<�I�N�U�Q�I�<�0 Z W ` ] M %  i F S U = L 8 C f Z ; < 2 8 ) $ T L f @ 2 " b [ c  Z 7   = Z V 9 Q V @ C , 5 F 8 L < B }  E f > F � G +  z � / # = F G \ R X @ h c D�k  -  [  �  �  ^  �  �  �  +  S    �  �  :  �  �  ^  �  _  �  �  #    �  �  �  �  �  ;  e  �  �    �  F  �    �  +  7  ^  9  l  u  n  �    �  u  �  �  `  h  �  �  �  �  �  y    r  \  [  �  �  �  t  h  K  y  x  J  i  <  �  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  F8  �  �  �  w  ]  E  .    �  �  �    L    �  �  �  U  �    �  �  �  �  �  �  �  �  �  �  �  z  _  C  $    �  �  �  j  y  d  O  :  %    �  �  �  �  �  �  �  �  �  �  �  �  �  �  4  5  6  7  4  0  %    	  �  �  �  �  �  �  x  Q  (  �  �  C  L  N  J  =  .      �  �  �  z  =     �  �  6  �  �   �    ?  l  �  �  �  �  �  �  �  �  �  �  �  �  1  �  "  �  
  J  a  w  }  �  �  y  p  d  R  >  "  �  �  �  5  �  ]  &  �  L  N  ?  .      �  �  �  �  �  �  c    �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  j  _  T  H  =  �  D  ~  �  �  �  �  �  o  Q  .  	  �  �  x  *  �  �  �  O  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  \  p  v  |  x  s  h  \  M  :  #  
  �  �  �  �  j  F  #  �         �  �  �  �  �  �  �  �  �  �  z  i  Y  4    �  �  �  w  n  d  Z  Q  F  6  &      �  �  �  �  �  �  �  	  .  R  ;  B  E  C  B  B  @  I  V  h  h  W  4    �  �  �  I    �  \  \  \  Z  Y  P  D  9  -         G  [  F  /    �  �  �  �  q  c  U  D  2  !    �  �  �  �  �  �  p  ^  L  9  '        �  �  �  �  �  �  [  &  �  �  p  *  �  �  ;  �  �  H  M  2    �  �  �  �  �  �  u  V  6    �  }    �  !  �    "  -  8  B  O  V  W  Q  D  5  !    �  �  �  Y    �  �  K    .  /  -  /  0  B  P  ]  e  i  j  c  U  >    �  �  N   �  	�  
  
;  
B  
5  
  	�  	�  	  	[  	   �  n  �  <  |  �  �  �  �  �  �  �     �  �  �  �  �  �  �  �  e  <    �  �  y  -  �  �  z  t  m  g  a  [  U  O  I  D  >  8  0  "       �   �   �  �  �  �  �  x  a  K  5      �  �  �  �  �  �  �  �  �  �  *  $      �  �  �  �  �  �  �  d  8  
  �  �  p  1  �  z  �  �  �  �  �  �  �  �  �  �  �  |  s  h  ]  S  #   �   �   x  �      �  �  �  �  r  D    �  �  n  7  �  �  :  �  A  �  �  �    	  	  �  �  �  �  �  �  r  T  2  	  �  �  S     �  �  �  �  �  �  �  v  Y  =  "    �  �  w  C    �  �  G  
  !           �  �  �  �  �  �  �  �  �  �  �  �  q  a  P  �  �  �  �  �  �  �  �    Y  �  �  �  �  �  �  t  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  R  7     �   �   �   {    �  �  �  �  �  �  d  D    �  �  s  %  �  �  >  =  .    �  �  �  �  �  �  �  �  �  �  �  Z  !  �  �  C  �  �  H   �  l  �  �  �  �  �  �  �  r  B    �  X  �  g  �  3  �  �  �  �  �  �  �  �  �  �  �  o  P  ,    �  �  7  �  �  P    �    $      �  �  �  X    �  �  ]    �  �  ]    �  �  t  �  �  �  �  �  x  d  z  �  y  [  -  �  �  �  W  
  �  �  j  R  F  9  ,            �  �  �  �  �  �  d  F  $  �  Z  �  	   	�  	�  	�  	f  	  �  8  �  �  ~  '  �  a  �  1  �  �    �  �  �  �  m  I  )    �  �  �    �  �  �  �  ;  �  �  $  \  Y  U  Q  N  G  7  (      �  �  �  �  �  �  e  H  *    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  V  �  �  �  �  @  >  ;  8  6  2  *  "      	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  b  K  2      �  �    '    �            �  �  �  �  L    �  r  (  �  r    r  �  +    �  �  �  �  �  �  �  ~  g  M  0    �  �  �  \    �  �  �  �  �  �  �  �  �  �  �  �  z  g  O  1    �  �  �  L  �  j  �  �  �  �  �  g  E    �  �  �  t  D    �  �  "  �  L  
      	    �  �  �  �  �  m  G    �  �  m    m  A   )  '       �  �  �  �  �  �  q  V  :    �  �  �  g  5    �  4  @  I  U  a  Y  I  +    �  �  {  D    �  �  b    �  =  �  �  �  �  �  �  �  �  �  �  �  �  |  W    �  �  L     �  �  �  �  �  �  �  �  i  N  3    �  �  �  �  �  �  o  N  '  )  .  -  &          �  �  �  �  �  �  �  �  �  �  h  G  �  �  �  �  �  �  y  g  V  B  .      �  �  �  �  ]     �  �  �  �  �  �  �  �  y  o  e  Y  L  >  1  $    	   �   �   �  _  s  x  o  b  S  B  -      �  �  �  �  �  O  �  "  |  e  c  U  H  7  %    �  �  �  m  3  �  �  q  L  X  o  `  /  �  ]  �  �  �  �  �  �  �  �  �  ~  S  $  �  �  0  �  �  S  �  �  �  �  �  �  |  x  s  o  j  r  �  �  �  �  �  �  �    %  k  a  W  L  B  8  -      �  �  �  �  �  ~  f  M  5      |  y  z    ~  s  _  I  .    �  �  �  K    �  �  u    �  ,  E  L  L  A  1    �  �  �  n  8  �  �  N  �  :  �  �    �  �          �  �  �  t  H    �  �  .  �  /  �  �  F     	    �  �  �  �  y  <  �  �  :  �  �    6  �  �  �  �  _  p  v  l  \  J  7     �  �  �  �  g  0  �  �  V  �  z  �  %       �  �  �  �  �  x  ]  A  #    �  �  �  T     �   �  �  �  �  �  �  �  �  �  �  �  �  u  b  M  5      �  �  �  _  `  a  b  b  ^  Z  V  S  P  M  J  L  Q  V  \  �  �  �    %  $  #  !        �  �  �  �  �  �  k  Q  6       �  �  �  �  �  �  �  �  z  u  p  h  ]  P  A  1  !  	  �  M  �  j  �  �  �  �  �  �  �  �  z  N    �      �  ;  �  �  
  �  �  �  �  �  �  �  �  �  �  m  B    �  �  �  u  ?  $    �