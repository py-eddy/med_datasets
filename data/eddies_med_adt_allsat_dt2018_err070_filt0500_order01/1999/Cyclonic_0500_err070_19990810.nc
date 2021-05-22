CDF       
      obs    =   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��+J      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�^�   max       P�>�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���T   max       <�t�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=q   max       @F��\(��     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��G�z�    max       @vnz�G�     	�  *   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @O�           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @���          �  4   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���m   max       ;ě�      �  5   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��l   max       B4@�      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��[   max       B4s�      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >G�#   max       C�PB      �  7�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =�x�   max       C�P      �  8�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          b      �  9�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A      �  :�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =      �  ;�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�^�   max       P��      �  <�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���
=p�   max       ?��8�YJ�      �  =�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��{   max       <�t�      �  >�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=q   max       @F��\(��     	�  ?�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�    max       @vnz�G�     	�  I   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @O�           |  R�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @�           �  S   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  T   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?fOv_خ   max       ?�ԕ*�1       T�            D                              "   "         $               	      !   b   K      #   -   !   "   	   *            *                  C                  ;                        )      !   N���N���Np��P�>�N�]DO�	N�M�O0wtN,�OS�N�O�x�M�^�O���O��N��ZN5��O��2O%��N+��O~��Op�N��N���O�"P-��O�,N;��O�%HP �SP`̝O d)O�xO�X�N�aN�fN��O��eO@��OV�VN��3Ny��NO�$PV�kP$�O� �O�N{��O�;mO��JN�?�N��0NaHN2��N���Oڵ>O#�FOԇO�bO�Pq�<�t�<#�
;��
;o%   ��o��`B��`B�o�o�t��D���D���T���T���e`B��o��o��C����
��9X��9X��j��j�ě�������������P��P��������w�#�
�#�
�'0 Ž8Q�<j�<j�@��@��@��@��D���D���L�ͽP�`�]/�]/�e`B�m�h�q���u�u�y�#��%��7L���T���T�##�������3<AHTUW^[WURHB<83333�����������������������0n������{[I0��EO[]hntztsh[ROKHEEEE�������������������������


�������������������������������

������������������������������������������������������)15>@=5)�������

������������aUL></&��#<HUda)16=BOchpnh[O6)  ")����

���������� "$/9;B;3/)"        CHTaz�������zmdYRKHC��������������������
"#%#
T[gw�����������tcYUT&/<HU]a_ZZWUH<43/-)&@BCMO[\fhqrphh[QOIB@��������������������+09Ibkilkec`VI<1/,(+7<Uanvz~�����}aUH637����������������������������������������gt��������������|zsg��������������������Za����}��������~tg[Z������

������\ht����������{th\]\\��������������������PTZamz{zvmibaWTQQOPP��������������������}��������{}}}}}}}}}}
0<T^`XI5050#

!*/5;CDCDCA>;/"���������������������������������������������������������������##$$����������)6BA=6*���������������������������������������������
#,,#
��������������������-0<HUalwxusjaUH<6/--zz�������������}{zzz��������������������[[hqt���tmh[VT[[[[[[����������������������������������������Ngt������������WNECN��������������������'*3BN[gopkfXN:5)"#'y��������������|xxwy5Fg���������g[N/%���������������������ֺк̺ɺɺɺֺ���������ֺֺֺֺֺ�����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����������l�V�I�H�V�s�������������	�	��Y�M�W�Y�_�e�n�r�}�~�����~�{�r�e�Y�Y�Y�Y����|�l�g�l�s�|������������������������ìæàÚÜàæìùù��������������ùìì�"���	����	���"�/�6�;�B�L�H�;�/�"�~¦�:�3�-�!��!�#�%�$�-�:�F�S�]�U�S�N�F�=�:�ʼļʼϼּ����ּʼʼʼʼʼʼʼʼʼ���¿´²º¿����������� �� ������������²°®²·¿������¿²²²²²²²²²²�B�L�[�h�l�h�[�R�O�B�6���� �#�!�&�6�B�����������/�:�H�T�\�d�g�j�e�a�H�;�"���������������������������������������������������������
��
������������������
�� � ��
��#�<�I�U�[�\�Z�U�I�<�#��
���s�p�o�l�s�������������������������������������������Ǽʼͼʼ��������������������ܾ׾ҾѾ׾�����	���"� ��	�����
���������#�/�<�H�U�W�V�W�U�H�;�/��
�������������������������ʾ˾ɾ����������ѿƿĿ������������ĿѿԿݿ߿ݿݿѿѿѿѻû������������ûл����'�-�+����лù����l�a�b�k�x�ú�'�3�@�B�3�*����ù����L�B�K�W�e�~�������ɺκ̺ĺ��������~�e�L���������������ʾϾ˾ʾ������������������ûûʻɻûлܻ���4�I�@�=�0���ܻлüY�@�4�*�9�@�Y������ּ�����ټʼ����Y���������������{�}����������B�N�M�5�D�D�D�D�D�D�D�D�D�EEEE*EEEEEED��������}�x�w�x���������������������������y�m�`�Q�L�R�`�y�����Ŀ˿̿ʿ����������y���� ���������(�+�5�<�5�(����"�������"�.�;�C�G�N�G�;�.�"�"�"�"�C�:�6�6�6�C�O�Y�T�O�C�C�C�C�C�C�C�C�C�C�����{�|�����������ݽ��������н����������������������������������������������������������������(�'�#�!��������z�t�n�a�_�a�d�n�zÇÓÔ×ÓÇ��z�z�z�zŠŠŠśřŠŭŹżŹŸŭŭŠŠŠŠŠŠŠ�����������������������������������������������������!�G�l�������������}�G�!��ٽ����ɽ������Ľнݽ߽��-�4�&�����ٻ������������������ûлܻܻ���ܻػû����������������'�4�6�5�4�+�'�����ݿտѿοοѿݿ�����ݿݿݿݿݿݿݿ�ƎƁ�u�\�6�(�����*�C�O�h�uƁƒƛƝƎ����վǾþľʾ׾��	�"�-�3�0�(�"��	���������������������������þʾҾʾ�������Ç�}�~ÇÊÓßàãåáàÙÓÇÇÇÇÇÇ�����������¹ùĹʹϹҹԹϹù������������ֹܹܹݹ���� ������߹ܹܹܹܹܹܹܹܿ����������������������������������������-�����%��$�8�I�V�X�b�m�j�b�`�V�I�-�O�K�B�A�?�B�O�[�h�t�zāčďčĂ�t�h�[�O�������j�c�`�f�s������������������������¦ ®²�������
�����������¿¦�[�Y�d�h�v�yāĊĦĳĿ������ľĤėā�t�[����ļķĵ���������
�*�B�L�<�#�
�������� - ? 9 > M . / \ N 4 R  k \ ? 5 * 6 C 5 $ ; 7 U J x 7 U u T | D O F d ` B \ \  . y 4 B F ( & N m ' P ] ^ v Q N 9 j X  C    �  �  �  >     �  	  �  +  4  3  7  ?  �  �  �  K  �    N  �  �    �  �  �  =  s    X  a  u  H  �  
  �  &  ^  �  �  �  �  e  �  �  ,  $  �    �  �  �  �    �  )  y  y  �  O  `;ě�;�o�49X������㼓t���j��9X�e`B�e`B��o�o����@��<j��t���1�P�`�ě��ě��@��0 Žo���ͽe`B���m���ͽC���t����hs��t��@����
�8Q�@��@���1�q���}�Y��Y��T����l����-��t��}�ixս�C���l��}󶽉7L��O߽�7L������������񪽾vɽ�xս�S�B?TBO�B��B&��BP�B"sB�>B `B��B��Br�B��B��B<�BCqB/`A��lA�ףBBB$�3B	�BB�B5wB*s7B&��BT�Bd�B4@�BËB+��B@/B�B|�B�^A��rB9�B
��B%ЋA��jB�uBtB&tBi�B@�B�\Bh�B�IBGDB�pB�JBSB�B��B WABdB
'�BuB�8B)�B�B]~BB;B>#B��B&��BC�B"Z�B��B ��B��B��BA)B�B4�Bc�BA�BAA��[A�c�BZ�B$�GB	�=B<�B��B*]�B&�KB��B?nB4s�B�DB+�)B?�B�FB�SBAWA���B2rB
�B%J�A���B�mB<�B�gB6B HB?�BD�B��B�B�B@�B@%B�B��B ��B��B	ݼB4�B��B
��B	��B�@A# A���C�PBA���?�A�AH@�A� qA��:A��^@|} A7�A��^A�5�A�ZA�c�A��sA�W~A�9<A��@�dAYTyA�wAK�Ay��@��D>U�@3�AN�O@��@�tA�%>C�M�@�WfAp[SA�\[A`�B �A&��A��%A��AȘ1A�؋AI�Aw�A-��@�O�@�\)A}6;B{WAY
AL��A�d�>G�#?xEAr��B>A�?OA�]dA���A߮A�2@@�A��SC�PA��?���AGA̓AA���A��~@z��A uA�z:A��TA�w!A�g#A�
A��A��A�yr@�VhAZ߬A��~AJ�>Ay��@��=�x�@;�ANh�@���@��%A��DC�GL@�Ar��A��rA`z�B ǜA#�A���A��:A���A�1AH��A�A1 @���@��A~	$B7AY�AJ�A�l�>@3�?.��AryB@�A��EA�u�A�wpA�r�A�9�            D                              #   "         $               
      !   b   K      $   .   "   #   
   *            *                  C                  <   	         	            )      "               A                                  !                              #   3   #      '   /   =         !            '                  1   '            #                     !      %   !   /   %            5                                                                   '            /   =         !            '                     '            #                     !      #   !   )   #N���N���Np��P��N�]DO,��N�HO�XN,�OS�N�Oo��M�^�O���Oe=^N��ZN5��O���O%��N+��O�wN�fN��N���O���O��O�)N;��O-�$P �SP`̝N�:
O�xO�X�N�aN�fN��O��eO@��N��;N��3Ny��NO�$OH3�P$�O}�N���N{��O�;mOn�N�?�N��0NaHN2��N���Oڵ>O#�FO���O�bO��sO�r�  w  /  �  �  �  �  {  $  �    H  0  �  v  �  (  �  �  z  �  �  o  �  E  �  T  	�  �  �  1    
0    �  j  ,  �  �    U  M  �  �  I  y  I    U  R  f  V  �  C  '  �  R  g  �  R  �  s<�t�<#�
;��
�o%   ���
�49X�o�o�o�t��u�D���T����j�e`B��o��1��C����
��h�o��j��j��h�D���D�����<j��P���<j����w�#�
�#�
�'0 Ž8Q�]/�<j�@��@���1�@��L�ͽH�9�L�ͽP�`��7L�]/�e`B�m�h�q���u�u�y�#��7L��7L���置{�##�������3<AHTUW^[WURHB<83333��������������������#0b������{UI0 EO[]hntztsh[ROKHEEEE��������������������������	����������������������������������

������������������������������������������������������)+25774)�������

������������aUL></&��#<HUda(,6BOV[`bb[HB62)'%'(����

���������� "$/9;B;3/)"        PTamz�������zmaVSOOP��������������������
"#%#
_gnt���������tig^[__7<HHUUXXUUHD<6217777@BCMO[\fhqrphh[QOIB@��������������������.5<IU_bgf\WUI<520,,.9<HUanqtx{|zwnaUN;99��������������������������������������������������������������������������������Za����}��������~tg[Z����

����������\ht����������{th\]\\��������������������PTZamz{zvmibaWTQQOPP��������������������}��������{}}}}}}}}}}
0<T^`XI5050#

!*/5;CDCDCA>;/"����������������������������������������������������������������������������������)6BA=6*���������������������������������������������
#,,#
��������������������<CHUabgjjga_UH<4114<zz�������������}{zzz��������������������[[hqt���tmh[VT[[[[[[����������������������������������������Ngt������������WNECN��������������������),5BN[gklkhcVN5)&&')y��������������|xxwy")5Bg��������tg[N1'"���������������������ֺк̺ɺɺɺֺ���������ֺֺֺֺֺ�����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����u�d�^�S�V�e�s������������������������Y�M�W�Y�_�e�n�r�}�~�����~�{�r�e�Y�Y�Y�Y������s�r�k�s�������������������������ìëàÞàâìù��������ùðìììììì�"���	����	���"�/�5�;�@�J�H�;�/�"�~¦�:�3�-�!��!�#�%�$�-�:�F�S�]�U�S�N�F�=�:�ʼļʼϼּ����ּʼʼʼʼʼʼʼʼʼ���¿·µ½¿���������������������������²°®²·¿������¿²²²²²²²²²²�B�L�[�h�l�h�[�R�O�B�6���� �#�!�&�6�B�/�"����"�1�;�H�T�Y�a�b�b�a�W�T�H�;�/��������������������������������������������������������
��
������������������
�������#�0�<�H�U�W�V�L�<�0�#��
���s�p�o�l�s�������������������������������������������Ǽʼͼʼ����������������������������	�������	������
��
�
���#�/�<�=�>�<�6�/�#��
�
�
�
�������������������������ʾ˾ɾ����������ѿƿĿ������������ĿѿԿݿ߿ݿݿѿѿѿѻû������������ûܼ���%�$�����ܻлù������y�s�}�����ùܹ��������ù����l�e�Z�Y�^�e�m�~���������úº��������~�l���������������ʾϾ˾ʾ��������������������ܻջллܻ������'�3�(�������Y�@�4�*�9�@�Y������ּ�����ټʼ����Y���������������{�}����������B�N�M�5�D�D�D�D�D�D�D�EEEEEEE
ED�D�D�D�D컅�����}�x�w�x���������������������������y�m�`�Q�L�R�`�y�����Ŀ˿̿ʿ����������y���� ���������(�+�5�<�5�(����"�������"�.�;�C�G�N�G�;�.�"�"�"�"�C�:�6�6�6�C�O�Y�T�O�C�C�C�C�C�C�C�C�C�C�����{�|�����������ݽ��������н�����������������������������������������������������������������������z�t�n�a�_�a�d�n�zÇÓÔ×ÓÇ��z�z�z�zŠŠŠśřŠŭŹżŹŸŭŭŠŠŠŠŠŠŠ�����������������������������������������������!�.�:�G�I�T�X�V�G�?�:�.�!��ٽ����ɽ������Ľнݽ߽��-�4�&�����ٻ������������������û˻лٻܻ߻ܻջû������������������'�4�5�4�4�*�'�����ݿտѿοοѿݿ�����ݿݿݿݿݿݿݿ�ƎƁ�u�\�6�(�����*�C�O�h�uƁƒƛƝƎ��׾Ѿ;־׾���	���"�*�)�!��	����ྱ�����������������������þʾҾʾ�������Ç�}�~ÇÊÓßàãåáàÙÓÇÇÇÇÇÇ�����������¹ùĹʹϹҹԹϹù������������ֹܹܹݹ���� ������߹ܹܹܹܹܹܹܹܿ����������������������������������������-�����%��$�8�I�V�X�b�m�j�b�`�V�I�-�O�K�B�A�?�B�O�[�h�t�zāčďčĂ�t�h�[�O������n�f�d�m�s������������������������¦ ®²�������
�����������¿¦�h�a�f�g�x�zāčĦĿ��������Ļģĕā�t�h��������ĿĹ���������
�#�'�>�B�=�0��
�� - ? 9 : M * 7 [ N 4 R  k \ - 5 * , C 5  0 7 U G ] 0 U ^ T | E O F d ` B \ \ 6 . y 4  F - ( N m ( P ] ^ v Q N 9 ` X ~ L    �  �  �  p     g  �  ~  +  4  3  �  ?  �  �  �  K  +    N  9  �    �  �  }  E  s  �  X  a    H  �  
  �  &  ^  �  �  �  �  e  �  �       �    �  �  �  �    �  )  y  �  �  �  �  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  w  u  s  o  h  _  S  D  2    	  �  �  �  �  �  w  \  \  q  /  %        �  �  �  �  �  �  �  �  �  j  M  0     �   �  �  �  P    �  �  �  ^  .  �  �  v  .  �  �  2  �  l    �  H  �  �  �  �  �  �  R    �  �  P    �  I  �  �  j  �    �  �  y  L      �  �  �  �  ~  F    �  �  k  5  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  c  ;    �  �  ?  n  t  x  {  {  z  s  b  O  9     �  �  �  v  ?    �  z  4    !  #      �  �  �  �  n  U  ;    �  �  v  4  �  �  �  �  �  ~  s  l  d  ]  T  K  A  7  +         �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  u  e  R  ?  +        �  H  1      �  �  �  �  �  �  �  �  �  �  �  �  }  i  T  ?  "  )  .  /  (    	  �  �  �  �  �  y  L    �  v  �  �    �  �  �  �  �  �  �  �  u  P  )    �  �  �  h  @    �  �  v  g  d  e  a  d  i  p  o  s  _  +  �  �  +  �  1  �  �  y  4  c  �  �  �  �  �  �  �  ~  ]  5    �  �  u  )  �      (  '  &  %  $  "                    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  b  R  C  4  $      �  �  �  �  �  �  �  �  �  �  �  g  C    �  �  y  8  �  �  L  �  z  q  g  \  K  :  *      �  �  �  �  �  �  �  �  i  O  6  �  �  �  �  {  r  i  `  W  M  B  3  %      �  �  �  ^  2  b  �  �  �  �  �  �  �  �  �  �  l  =    �  |  3  �  �  f  9  #      &  A  S  `  j  o  a  0  �  �  c    �  Y  [  S  �  �  �  �  �  �  �  �  �  �  t  f  V  G  7  "  �  �  �  �  E  B  ?  ;  8  5  1  .  +  (  "      	     �   �   �   �   �  �  �  �  �  �  �  �  �  m  P  1    �  �  �  -  �  8  �  {  �  �    4  S  O  1  �  �  c    �  6  
�  	�  	     (    �  	  	<  	b  	�  	�  	�  	�  	�  	~  	?  �  |    �  &  �  4  R    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  j  ]  P  D  �  =  L  `  z  �  {  _  8  
  �  �  X  Q    �    <  \  k  1    �  �  �  �  ]  3    �  �  _    �  }  *  �  r  �   �    �  �  �  �  x  r  ~  m    o  K    �  �    �  �  p  t  	�  	�  

  
'  
.  
/  
  	�  	�  	{  	*  �  Z  �  W  �  -    �      	     �  �  �  �  �  �  �  �  �  �  �  ~  l  R  1  �  w  �  �  �  �  |  R  !  �  �  Q    �  �  F  �  �  y  :  �  C  j  c  [  S  K  D  <  4  *      �  �  �  �  �  s  1   �   �  ,  $         �  �  �  �  �  �  g  B    �  �  �  �  w  c  �  �  �  �  �  �  �  �  �  �  �  Z  2  	  �  �  �  Q      �  �  �  �  r  O  +    1  %    �  �  �  �  5  �  k  �  h  �        �  �  �  �  �  l  P  1    �  �  c  +  �  �  a  9  <  <  :  8  :  @  G  M  S  T  O  @  &    �  �  �  $  �   �  M  :  '    �  �  �  �  x  R  +    �  �  �  z  T  +     �  �  �  �  �  �  �  �  �  �  �  �  �    d  -  �  �  �  |  Y  �  �  �  �  �  �  �  �  �  �  ~  q  \  C  *    �  �  �  }  �  +  ;  ,       %  !    *  >  G  )  �  y  �  >  �  �    y  s  d  E  :  <      �  �  �  g  3    �  �  q  �   �     C  H  I  E  ?  7  -      �  �  �  Q  %  �  �  n  �  G   �            �  �  �  �  �  |  _  @  !    �  �  �  �  u  U  L  C  8  +      �  �  �  �  �  �  �  �  �  �  �  y  o  R  C  7  !    �  �  �  �  l  Q  ;     �  �  �  ^    �  #  �    O  _  f  e  _  U  G  0    �  �  S  �  {  �  :  �     V  D  2      �  �  �  �  �  �  �  �  �  �  �  r  d  _  [  �  �  �  �  �  e  D  !  �  �  �  y  E    �  �  |  J    �  C  B  @  =  7  /      �  �  �  v  E    �  �  w  >  �  8  '  ?  W  \  V  O  D  :  .  #        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  j  X  C  -    
  �  �  �  R  <  %    �  �  �  �  c  ;    �  �  �  b  ,  �  �  ?   �  g  I  *    �  �  �  �  w  R  )  �  �  �  c  /  �  �  8  U  �  �  �  �  �  �  {  \  B     �  �  �  [    �  G  �  ]  �  R  6    �  �  �  k  :  	  �  �  m  -  �  x    �    �  A  �  �  �  v  `     �  �  ]  &  �  �  �  V    �  K  �  t  �  ]  m  s  r  j  T  9    �  �  �  �  M    �  �    k  �   �