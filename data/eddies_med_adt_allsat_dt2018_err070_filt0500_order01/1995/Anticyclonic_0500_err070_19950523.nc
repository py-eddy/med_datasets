CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�n��O�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��~   max       PnB/      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �C�   max       >	7L      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @FFffffg     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vup��
>     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P�           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       >8Q�      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�\   max       B1O�      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�vY   max       B1��      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?���   max       C�y=      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?���   max       C�yj      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��~   max       P%�%      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���҈�   max       ?�<�쿱\      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �C�   max       >	7L      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�ffffg   max       @FFffffg     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�    max       @vt          �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P�           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��@          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���vȴ:   max       ?�6��C-     �  QX         �      #   
                        ;         F      	   1   	         2         R   8                  T   <   "   �   /                  <   (      	                              #NU��M��~P U�OP[qPUN�MBN.,�N��lN��-NN��OR@�O�H~NN�5P#�YN�ړO�`�PnB/N��N|�aP_>N��N�	FN�7YPofN� .O�PO��P"�Om�O0JjOK�OCɶOfpP\�O�MO���O�&,OD��NLb�O[�N��;N-Nh�WP-ѿO��O,��NŔ�N7E+O}��N�QPN:E�N�/OJjbN�e�N�uzNHmOk�N�C��u�o�ě���o�D���o�o��o:�o:�o;�o;�o;�`B<o<t�<t�<#�
<D��<T��<T��<T��<e`B<e`B<�o<��
<�9X<�j<�j<�j<ě�<���<�/<�/<��<��<��=o=\)=\)=\)=t�=�P=�w=,1=0 �=0 �=@�=u=y�#=y�#=��=���=��
=��
=��T>	7L��������������������%$!22:>F[gt�������[NB82PKRUanz���������znZP{uv����������������{ "),26BLOQOIB96)    1+.6BGGB<61111111111��������������������VZ`ainsz}��znaVVVVVV	
&)+) 								����������������������)5BKNTURJB5)���^\fhhtvz{tih^^^^^^^^������"2FMLHA)����������������������������������������������/771)
���������������������������z���������������������)5[hlZE5)����� #'$kt��������������{tk��������������������ssw���������������zs���&),)$ �����������������������������
$,*#�������������������������������������������������KIOTZ[hty}|wtoh[ROK����
#/00-'#
��!"/;GHJHD;1/"�������������������������)5BNUWUNF5)��002;@HMTeipmgaTH;6/0226B[gt}���tg[NJC>2�������
""
���������������������������������������������#*/7<?DD></$#713<HTUYVUH<77777777_akmz�����{zma______FHIHE?HUZ\acaUPHFFFF�����'./+�������)5>BEFDB5)�	
#0<70-'"
!#&-02<AE?><40)#@CEOSZ\g_\[OHC@@@@@@������������������������������������*055*����

 ����������������� ����������~{~�������������������������������������������������������������#%!	�����I�U�V�b�`�V�I�F�@�A�I�I�I�I�I�I�I�I�I�I�.�:�;�G�H�G�:�.�+�-�.�.�.�.�.�.�.�.�.�.�)�B�h�tāĎČą�t�G�;�6�)�#������)E�E�E�E�FFFFE�E�E�E�E�E�E�E�E�E�E�E���������������������������������������x�������������������������x�v�m�x�x�x�x�zÇÓÙßÓÇÀ�z�t�z�z�z�z�z�z�z�z�z�z�'�3�>�@�A�@�5�3�'���������
���!�'���������������������������������������ſ����������������y�t�y�~��������������������� �-�:�F�R�Q�F�:�-�!��������������9�D�G�G�A�:�(������տ׿�������	�����	���������������������������;�G�y�������������y�m�`�G�;�.����.�;�ݽ����������������ݽѽԽݽ��N�Z�g�s�������������������g�Z�H�B�>�A�N�/�H�a�}�����m�T�H�/�'�&��	�����������/�������������������������������������T�a�i�m�u�v�m�a�T�S�R�T�T�T�T�T�T�T�T�T�<�U�l�|Łńŀ�^�U�I�0�
�����
����+�<�����������������������������������������<�>�H�I�Q�N�H�B�<�0�/�#�!� �#�#�+�/�9�<ƧƳ������ƺƵƳƧƚƕƔƚƦƧƧƧƧƧƧ���(�A�g�u����|�s�g�Z�A�(��	�� �������������������������������������������	��"�.�.�5�;�>�=�;�.�"����	�	���	�"�/�9�9�7�/�"�	���������������������"�������)�6�M�B�,��������ëâáìó���tāčĚĢħğĞġĚčĉā�t�h�b�[�\�h�t�-�:�F�S�_�a�k�e�_�S�F�:�-�'�!���!�%�-�`�m�y�����������y�m�`�T�G�C�?�D�G�T�Y�`��"�/�;�H�H�M�T�H�F�;�/�"��	���	�����*�?�C�O�S�O�G�C�6�*���������\�uƧƳ�����������ƳƚƁ�u�g�a�Y�S�\����0�E�I�T�I�D�#�����Ŀĸķĸ���������g�t�|�w�o�g�[�N�B�5�)�#�!�&�)�5�B�gD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D~DwDtD{D�àìù����������ûìàÜÓÇÅÀ�~ÇÓà�������������������~�|�������������������������s�f�Z�U�O�Z�[�f�s�}�����������������������������������������ŔŕŠŢŨŭŭŭŠşŔœőŐŔŔŔŔŔŔÓÞàáìùþþùìãàÞ×ÓÌÓÓÓÓ������"�8�<�:�0�"��	����������������������������������������¿«­³¿������Z�f�s�����������������������s�f�N�P�Z�������������Ƽ���������������}��������m�u�y�}�y�m�k�`�T�S�T�[�`�g�m�m�m�m�m�m�
��#�*�0�.�#��
���������������������
�y���������������������y�x�u�o�l�i�l�u�y�l�y���������y�l�j�k�l�l�l�l�l�l�l�l�l�l�\�h�u�zƁƃƁ�u�h�\�O�F�O�R�\�\�\�\�\�\EiEuE�E�E�E�E�E�E�E�E�EzEuEiE^ETEWE\EbEi�4�@�M�Y�f�h�o�h�f�Y�M�@�4�(�(�1�4�4�4�4���"���������������������ּټ������߼ּҼԼֺּּּּּּּY�e�r�~�����������~�r�e�Y�L�@�3�5�@�L�Y V L = 6 T ] 8 i F + [ E O / \ = L 7 D 3 , S U J X Y : H B 1  /  B [ 5 : D B F 5 � S 6 * ` 7 _ E m 5 ? " 4 N j J    �      �  �  �  G  �  �  e  �  o  `  �        �  �  �  �  5  �  �  �  r  �  �  �  s  �  �  �  �  �  �  �  �  z  B  �  �  }    3  �  �  k    V  M  �  �    �  Y  ���T��>\)<T��<��;ě�;ě�<t�;ě�;�o<ě�=\);�`B=��<u<�=��w<�o<�1=}�<�1=o<�t�=�%<�9X<���=��=��w=8Q�=<j='�=D��=49X=�G�=�^5=�o>8Q�=��w='�=<j=@�=��=#�
=���=��T=ix�=P�`=H�9=���=��P=��=� �=�
==\=���=�1>,1B$eB�rB	�B��Be�B��B�B ڂB��B�~B d{B�1B�ZB�B ��B�B\�B��B �$B&BٌB��B�]BLjB=B!�=B��B�[B��B��B(A�\B��B��A�-B�~B0B"!uB�6B�B�A���B��BA�B��B$�)B%�-B1O�BrB,��B/��B��B�B8MB�JB��BڶB�B�B	 �B�B��B�.B�5B �CB�oB�1B �B�FB�tB�B ��B��BC�B~B ��B@cBͣB�,B��B�LB�qB!��B��B�WBHB��B?sA�vYB��BÝA�{5B�+B@B"@wB@�BC�B�JA��B�QB��BĵB$�PB%��B1��B�B,��B/��B�B 7B?�B9iB֗B�BBsA�LA�Q�C�y=A�'$@�~yA��'?���A�c�Ao��@g0iA�y�A���Af��A/��A�r�A��B��A���A��A��IA��B�fA�E@���A_AtA���A���A�˴@�Aj"�A�F+A��B�A�&A��VC���A��AG5XAC��Aѓ3A�A��A��AA��2AD�@��HAi�3A�O�A�eA3B)gC���@�{�@��A�?�1�B7�AԼA�~{C�yjA�w�@���Aɀ�?���A���Ao�3@c��A�{�A��|Af��A1 RA�R�A���B߄A��8A�j�A��A'B�5A��k@�uA^��A�l�A�|Aݙ-@~>Ai��A�FTA��BA慇A��C��A�mIAF��AB��AэPA�{A�BA���A��ABX�@��Ai  A���ApA�<B@�C��@��E@��DA��?�1�         �      $   
   	                      ;         F      
   2   	         2         R   8                  T   =   "   �   /                  =   (      	                              #         +      '                     %      '      !   3         1            '         /   '                  /   %      #                     /                                                      #                                 !   %         )                        %                  !                              -                                       NU��M��~OS�#N�9GO���N�MBN.,�N9��N��-NN��OR@�O[/
NN�5O��gN�ړO�`�O��PN��N|�aP%�%N��N�m�N�7YOSPN� .O�Oi�>P��O-�iO&^�OK�O!�cOfpO�U�ODc�O��Og9�N�=<NLb�O[�Nn�fN-Nh�WP$`�OG�GO,��NŔ�N7E+OQY<N�]_N:E�N�/OJjbN�e�N�uzNHmOk�N  �  ;  �    &  M    �  �  �  h  �  �  A  3  �  Z  �  �  �  w  �  P  T  Y  �    �  �  �  �  �  O  Y  	L  �  �  Z  \    �  7  P  �  p  �  �  t  8  �  &  Z  �  �  �    ��C��u=y�#:�o;D���D���o:�o��o:�o:�o<�o;�o<��<o<t�=t�<#�
<D��<�j<T��<�t�<e`B=\)<�o<��
=��<�`B<�/<ě�<ě�<�h<�/=Y�=ix�=+=���=49X=\)=\)=t�=t�=�P='�=P�`=0 �=0 �=@�=�%=�o=y�#=��=���=��
=��
=��T>	7L��������������������%$!HDDGLN[gtt|���|tg[NHTUZafnz~~~{zngaYUTT|������������������| "),26BLOQOIB96)    1+.6BGGB<61111111111��������������������VZ`ainsz}��znaVVVVVV	
&)+) 								��������������������)5;BEIIFBA5)^\fhhtvz{tih^^^^^^^^����-5:=;85.) ������������������������������������������������	����������������������������z���������������������)5MWZXKB5)����� #'$���������������������������������������������������������������&),)$ ��������������������������������
�����������������������������������������������LIOPU[[htx}~|tmh[TOL����
#/00-'#
��"*/;AHFB;:/"��������������������)5BGLJG@5):::;<CHT]adefcaTHB<:46;BN[gt{���}tg[MD?4�������

���������������������������������������������#*/7<?DD></$#824<HQUXURH<88888888_akmz�����{zma______FHIHE?HUZ\acaUPHFFFF������-.*�����)57?BCBA;5)	
#0<70-'"
!#&-02<AE?><40)#@CEOSZ\g_\[OHC@@@@@@����������������������������������������*055*����

 ����������������� ����������~{~�������������������������������������������������������������#%!	�����I�U�V�b�`�V�I�F�@�A�I�I�I�I�I�I�I�I�I�I�.�:�;�G�H�G�:�.�+�-�.�.�.�.�.�.�.�.�.�.�6�B�O�[�h�k�s�r�h�h�[�O�B�>�6�/�+�*�.�6E�E�E�E�E�FFE�E�E�E�E�E�E�E�E�E�E�E�E����������������������������������������Żx�������������������������x�v�m�x�x�x�x�zÇÓÙßÓÇÀ�z�t�z�z�z�z�z�z�z�z�z�z�'�3�4�=�3�2�'�������&�'�'�'�'�'�'���������������������������������������ſ����������������y�t�y�~��������������������� �-�:�F�R�Q�F�:�-�!���������������(�)�4�7�5�/�(����������������	�����	���������������������������G�T�`�p�z�y�v�m�f�`�T�G�;�7�.�*�&�,�;�G�ݽ����������������ݽѽԽݽ��N�Z�g�s�������������������g�Z�H�B�>�A�N��"�/�H�V�`�n�o�j�a�T�H�/�"����������������������������������������������T�a�i�m�u�v�m�a�T�S�R�T�T�T�T�T�T�T�T�T�<�U�c�u�z�}�y�b�U�I�<�#�������#�<�����������������������������������������<�<�H�L�I�H�?�<�/�&�%�(�/�;�<�<�<�<�<�<ƧƳ������ƺƵƳƧƚƕƔƚƦƧƧƧƧƧƧ��(�<�N�W�c�c�a�Z�N�A�5�/�(�!����������������������������������������������	��"�.�.�5�;�>�=�;�.�"����	�	���	�	�� �"�'�(�%�"���	�����������������	��������)�6�B�1�"�������ïåäìø���tāčĔĚğĤěĚčā�}�t�h�h�f�b�h�t�t�-�:�D�F�S�_�`�j�d�_�S�F�:�(�!���!�'�-�`�m�y�����������y�m�`�T�G�C�?�D�G�T�Y�`�"�/�5�;�E�H�J�N�H�;�/�"���	�
���"�"���*�?�C�O�S�O�G�C�6�*��������ƚƧ��������������ƳƚƎƁ�w�o�n�q�uƁƚ�����
���#�(�$�����������������������g�t�{�v�m�g�[�N�B�5�+�$�"�)�5�B�gD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ùù��������ùòìàÓÇÇÅÇÒÓàìù�������������������~�|�������������������������s�f�Z�U�O�Z�[�f�s�}�����������������������������������������ŔŕŠŢŨŭŭŭŠşŔœőŐŔŔŔŔŔŔÓÞàáìùþþùìãàÞ×ÓÌÓÓÓÓ��������"�7�;�8�.��	������������������������������� ��������������¿¶¶¾���ؾZ�f�s�����������������������s�f�N�P�Z�������������Ƽ���������������}��������m�u�y�}�y�m�k�`�T�S�T�[�`�g�m�m�m�m�m�m�����
����#�'�-�+�#��
��������������y�����������������y�r�p�n�x�y�y�y�y�y�y�l�y���������y�l�j�k�l�l�l�l�l�l�l�l�l�l�\�h�u�zƁƃƁ�u�h�\�O�F�O�R�\�\�\�\�\�\EiEuE�E�E�E�E�E�E�E�E�EzEuEiE^ETEWE\EbEi�4�@�M�Y�f�h�o�h�f�Y�M�@�4�(�(�1�4�4�4�4���"���������������������ּټ������߼ּҼԼֺּּּּּּּY�e�r�~�����������~�r�e�Y�L�@�3�5�@�L�Y V L  , W ] 8 f F + [ " O % \ = Q 7 D 2 , F U @ X Y  D * 2  1  6 M 2 2 C B F 2 � S 5  ` 7 _ E ` 5 ? " 4 N j J    �    �  �    �  G  i  �  e  �  �  `  '        �  �  �  �  �  �  �  �  r  �  �  r  d  �  X  �  )  �  u  �    z  B  v  �  }  �  �  �  �  k  �  �  M  �  �    �  Y  �  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  f  U  C  2  !  ;  :  :  :  9  9  8  8  7  7  6  3  1  .  ,  )  '  $  "    
  g  �  �  t  )  �  [  �  �  �  �  @  �  <     �  $  	P  �  �  �  �  �  �  �          �  �  �  c    �  H  �  �  L  �  �    #  $    �  �  �  �  �  u  8  �  �  ,  �  �  �  ?  M  E  =  5  .  $      �  �    �  �  �  �  U    �  �  j                   �  �  �  �  �  �  �  e  ;    �  �  v  }  �  �  �  �  �  �  �  �  �  �  �  �  �  X    �  �    �  �  �  �  �  y  n  a  S  C  2      �  �  �  g  8     �  �  �  �  �  �  �  �  �  �  |  v  o  i  c  `  \  Y  U  Q  N  h  X  G  ;  +      �  �  �  �  �  �  �    q  R    �  �  �  W  u  �  �  �  �  �  �  �  u  \  :    �  ~    �  3  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  J  �  .  �  �  �     5  @  =  *    �  t    |  �  -  8  �  3  3  3  /  %        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  f  P  6    �  �  �  E    �  �  (  �  9  �  �    8  O  Z  W  F  .  
  �  �  �  d  �  �  �  �  �  �  �  �  �  �  �  �  w  j  [  I  8  %    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  N  )    �  �  d    �  f  Y  �  �  �  �  �  �  �  �  �  V    �  �  Y     �  �  h  �  w  r  n  e  [  P  G  =  4  *         �  �  �  �  I  �  �  T  �  �  �  �  �  �  �  �  �  �  p  O  )  �  �  �  �  r  =  P  O  N  M  L  I  ?  5  +       �  �  �  �  �  s  L  $   �  �  �  �         '  <  R  I  2    �  �  a  �  U  �  D  �  Y  L  @  4  (        �  �  �  �  �  �  }  j  W  L  A  6  �  �  �  �  �  �  �  �  �  �  �  z  j  V  >  %  	   �   �   �    �  �  9  k  �  �  �  �        �  �  2  �  &  V  J  `  �  �  �  �  �  �  �  �  �  q  6  �  �  N  �  x  �  A  "  �  �  �  �  �  �  k  L  $  �  �  �  Z  -    �  �  P  �  �  �  �  �  �  �  �  �  �  q  V  8    �  �  ~  -  �  R  �  �  2  �  �  �  �  �  �  �  �    b  D  $  �  �  �  a  6    �  �  �  �  �  �  �  �  �  ~  Z  1    �  �  t  +  �  �  /  �  �  O  M  C  4  "    �  �  �  �  �  u  P  '  �  �  �  h  =    �  {  �    @  U  X  N  ?  1    �  �  d  �  m  �  �  �  P  �  %  �  �  �  	  	-  	B  	L  	D  	  �  w  �  |  �  a  �  x  �  �  �  �  �  �  �  �  i  K  )    �  �  �  @  �  �  ]  �  <  �  �  �    }  �  �  �  Q  �  v  �  u  �    �  �  	�  %     �  �    4  N  Y  R  >    �  �  e    �     o  x  �  �  �  \  c  i  p  m  i  e  W  E  3    �  �  �  �  h  A     �   �      �  �  �  �  �  �  �  �  �  �  �  r  Y  )  �  �  M   �  }  �  �  �    w  i  T  <       �  �  �  �  v  \  E  /    7  6  5  4  3  2  1  0  /  .  #    �  �  �  �  �  �  {  f  P  A  1  "      �  �  �  �  �  �  �  �  �  �  �    q  c  �  �  �  x  b  �  t  H    �  �  j    �  n  �  ^  �  �  J    B  `  l  o  n  b  H  '  �  �  �  Y    �  k  �  }  �  �  �  �  q  V  @  3  $      �  �  �  �  �  �  z  K    �  �  �  �  �  �  �  �  �  �  n  X  D  0        �  �  �  �  �  t  k  c  Z  R  J  A  9  0  (          �   �   �   �   �   �   �  %  -  4  8  7  3  ,      �  �  |  F    �  e  )  �  �  �  C  Z  p  �  �  �  }  m  Z  E  -    �  �  �  O  �  m     �  &             �  �  �  �  �  �  �  �  n  R  5     �   �  Z  >  "    �  �  �  �  d  >    �  �  �  O    �  �  V    �  �  �  �  �  m  R  6    �  �  �  m  &  �  h  �  �  %  �  �  �  �  �  |  g  Q  8    �  �  �  X  �  �  G  �  �  e  5  �  �  �  v  �  �  U  #  �  �  ~  A    �  �  H  �  �  T  �        	  
  
       �  �  �  �  �  �  u  U  4    �  �  �  �  y  a  @    �  �  l  +  �  �  I    �  P  �  &  �  �