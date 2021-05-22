CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�-V�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�qy   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �D��   max       >+      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?W
=p��   max       @E7
=p��     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�X    max       @vP��
=p     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @O            l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ʩ        max       @�-�          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��w   max       >F��      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�1�   max       B(��      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��.   max       B(<�      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?H#:   max       C�`C      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?Nb�   max       C�ZI      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�qy   max       P8�d      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�ȴ9Xc   max       ?�H���      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �D��   max       >+      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?fffffg   max       @E7
=p��     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���
=p    max       @vP��
=p     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @O            l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ʩ        max       @�H�          �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��)^�	   max       ?�H���     �  N�   	         �   T      #      +   	      3   Z                           
            2         |      &   r      5          #                                    	      A      G   ?NHe�M�qyN$��P|�P��M���O?�OמO��N��tNo^�P nP>��O���O�OyGO]��NMO�N��0Nr�N��OWpPVCOK�RO�%N#�O6\Pq�vO�V�O�;�PS�N�P`EN�K�N���O�|�Nd	�N��N��O.��OM�O��O��O
�Nw�O0z O�Nj\KO)�O��O�IOʂ8O"Ѷ�D���ě�����T���T���o%   :�o;�o<t�<D��<D��<D��<T��<T��<u<�1<�9X<�9X<���<�`B<�h<�h<��=\)=�P=�P=�P=��=�w=#�
=,1=49X=8Q�=@�=D��=L��=L��=T��=m�h=q��=q��=u=y�#=�O�=�O�=�\)=�hs=�1=�1=� �=� �=Ƨ�>+qtw�������ytqqqqqqqq��������������������;2<HSPIHG<;;;;;;;;;;�������
! ��������/<awx}ym[H</()6=BIHBA60)((((((((?;7:BEN[dgryvtig[NB?UOUam�����������zqcU�����������������������������������������������������
)/<=HQTRRM</#�'%0<Hnz������nK><5/'A?>Haz��������zmaTLA�����

�������#/<AHOU\_UH</-&&# &*5BJNS[]`a[NB5)%@6=BHOPROB@@@@@@@@@@KKO[ht}�����th[WRPPK������������������������������������������������������������������������������)5BL[t}tgB5)�������������������������).=GF=5)����������������������gbdgmpy{|�����{xvng��)BOhz|thOB)����������������qliimt������������q��)5BN[bji[B)�������������������������)?IOQRNB)��������������������.)+/3<HRUSMH</......����������������
!#(&#
������������������������

 ��������������������������������������������������������������������������������������������)5=BCBB5,)��
#0;<@=<0,
��[UU[afmrz����zsmea[[;ABMNOZ[\\[NLB;;;;;;gntv����������~wqkgg�����������'),.-+)#	mea^_es�����������zm�����������������zÃÇÌÇÆ�z�q�n�k�n�r�z�z�z�z�z�z�z�z��������������������������������������������������������������������������������������%�7�8�5�&���������������������������"�3�;�:�;�/��������������}������ÓÔÕÓÑÇ�z�z�zÀÇÓÓÓÓÓÓÓÓÓ�����������������������������������������s������������s�f�Z�M�A�;�<�6�6�A�M�Z�s�����'�6�8�3�'����ܹϹ̹¹��ùϹ����������������������ŹŭŬūŭůŹ�����ƿ��������ĿοĿ��������������������������m�������������������y�m�`�G�;�4�3�7�T�m���A�U�]�i�e�N�A�5������ݿѿɿݿ���������������������������������������˻S�_�l�x�t�l�c�_�S�F�:�-�+�+�-�3�:�F�I�S��������������������������������{ŇŔŠūŮŠŝŔŋŇ�{�n�b�a�a�e�k�n�{���������������u�y�����������Z�f�q�i�h�b�Z�U�A�4�(�������(�M�Z���������"�����������������#�#�/�8�<�=�?�G�<�/�#�"����#�#�#�#�#�G�T�`�e�m�s�q�m�d�`�T�G�;�6�;�;�=�E�G�G��"�.�7�5�7�0�.�"��	������������	���f�u�u�y�����s�Z�J�A�������,�4�E�f�������ʾξ̾ʾ��������������������������`�m�y�����������������y�`�T�G�B�@�G�T�`�ûлڻѻлû��������ûûûûûûûûûýĽнݽ�������ݽнĽ����������������ļ���޼ռмļ��������~�Q�B�G�Y�����ݼ�����	��"�.�G�V�]�[�O�G�;�.�"�	���������(�5�A�Q�Z�g�r�g�a�Z�N�A��� �������)�B�k�|ā��t�d�O�����������������������������������������������������������$�0�I�K�H�A�7�$��������ƵƼ���������{ǈǔǡǫǭǺǭǫǡǔǈǁ�{�t�x�{�{�{�{D�EEEEE%E(EEED�D�D�D�D�D�D�D�D�D��<�H�U�h�q�t�|ÇîîàÕ�n�H��� �%�0�<������	������������������ÓàáìùþúùìàÓÎÈÎÓÓÓÓÓÓ�[�h�t�{�|�t�h�`�[�X�[�[�[�[�[�[�[�[�[�[������������������������������������������
��#�%�+�*�$�#��
�������������������A�N�Z�a�g�m�h�[�Z�N�G�A�@�8�5�2�2�5�=�A����������������������������������������²¿����������������������¿¸²«©¯²���������������������������������������s�����������������s�r�j�f�Z�W�X�Z�f�sŠŭŹ��������������ŹŭŠŗŔŒŔŖŠŠ��"�#�/�/�/�"�!��	� ��	��������������������������������������������������!�-�4�<�@�8�!�����ֺɺǺɺӺ����������Ǻ��������������~�z�r�l�k�r�~�������������ûлܻ�������ܻлƻ���������D{D�D�D�D�D�D�D�D�D�D�D{DqDoDlDkDoDoD{D{ A o T ' + w  _ & 8 | 0 8 ( + D / b = ? F 0 8 l 6 2 V w ` = = 7 b P E " s R : A , " + " h J <  D I L @ M .  d  /  /  
    V  �  f  �  �  �  `  x  �  <  e  �    X    �    J  	  �  T  8  �  �  3  X  �  �  �  �  �  �  v  �  6  s  �  C  B  x  �  �  B  �  �  t  h    a��w��1��o>o=�\);�o=\)<�h=@�<�C�<�C�=�%=��=#�
<�=C�=#�
<���=<j=\)=\)=��=�P=m�h=q��=� �=8Q�=D��>%�T=�o=��->�w=e`B=Ƨ�=��=���=�1=e`B=�o=�o=���=��-=��w=��T=�^5=��T=�E�=� �=�j=�`B>��=�l�>+>F��B
SlB"2B#�BFLB�B�*B�%B&B B�B��BZ\B��Bn[A�1�B#��B��B�BnB�B�tB��B��B"2B(�BHB�=B"�B(��B�@Ba[B
�qB/%B��Bx�B(B�BiB$�(B!�zBB�3B%SB$BD�B?B�LB%A�y�B9VB
p�B4B��BB�B}�B
@B"K�B/�B@�B�B�NB��BA�B�^B�kB
��B6�B�A��.B#��B�?B��BB�B�B|=B� B�kB"BlBPBC�B�bB"�3B(<�B�!B`�B
�B�B��B��BB�B��B��B$�)B"?rB~HB�BAKB�?BIB<�B<�B$��A�_pB?sB
��B>pB��B�B��AȘy@$hA��A��7A��fAɓMA�m%A@� ?H#:A��At�[Aj�A��A�[@���A�`qA��5AF�A8��?V�A�k�AgXXA]��A;3!AL�/Al�@�A&yn@��{A_��A�A�[�A�&B?mB?C�`CA�T7A�A��LA۷�A�gA�"�A�w�A��A�3�A��\AC�{A��'A��A�۫@\C�@p@�-C���AȂ�@$�A�t8A���A��nAɅA��ZAA5?O�iA��0AthkAkE*A��<A�|}@�axA�gCA�'AGZA8�>?Nb�A��Ag��A]�A?0aAMlAn��@��iA(�z@�A`�A���Aք�A��BBu�B�.C�ZIA�o�A�A�x�Aۼ
A�	A��A��)A�}�A��"A��jAB��A��A�{5A��@\:�@}�@���C��s   
         �   T      $      ,   	      4   Z                           
   	         2   	      }      &   s      5          #                                    	      A      G   ?            1   5         %   #         %   /                                 -      #         ;         3      '         +                                                "                  '            #            '                                 -      !         %         +      %         +                                                "   NHe�M�qyN$��O���P�@M���N��wO��lO�|�N�/�No^�O��P�O�^iO�N�@?O.uNMOt� N��0Nr�N��ROWpPVCO"�2O�_"N#�O6\O�M�O�V�O��P8�dN�O�;{N�K�N���O�|�Nd	�Nl5gN��O!A�OM�O��N�{;N��Nw�O0z O�Nj\KO)�O���O	��Oʂ8O"Ѷ  �  �  �  I  �  �    |  u  L  F  �  

  �  W  �  W  Z  �  �  �  �  _  �    =  �  �  �  �  �  �       j  	�  �  n  �  �  V    |        E  �    �  ]    �  ҽD���ě�����=y�#<u�o<o;�`B;��
<#�
<D��<���<��<��
<T��<�9X<ě�<�9X<���<���<�`B<��<�h<��=��=8Q�=�P=�P=��=�w=0 �=P�`=49X=]/=@�=D��=L��=L��=Y�=m�h=u=q��=u=�%=�t�=�O�=�\)=�hs=�1=�1=�9X=�9X=Ƨ�>+qtw�������ytqqqqqqqq��������������������;2<HSPIHG<;;;;;;;;;;����������������#/HUaipoj\H</#()6=BIHBA60)((((((((@=?BNT[\gmtttog[NB@@]\az�������������nb]�������������������������������������������������������
#/<AFLLH</#
�--38AHanz������]H<0-EDHTamz�����zmaUPIE�����

�������,++/<HITQH=<:/,,,,,,!#)+05BDNP[[^^[NB5)!@6=BHOPROB@@@@@@@@@@YURRRO[htx������th[Y������������������������������������������������������������������������������)5BL[t}tgB5)������������������������$19>A?5)������������������������gbdgmpy{|�����{xvng����6=MQPIB6)�������������rmkkot������������tr�� )5BN[cgf[)�����������������������):DIMNNKB5)
���������������������.)+/3<HRUSMH</......����������������
!#(&#
������������������������

 �������������������������������������������������������������������������������������������������)5=BCBB5,)��
#0;<@=<0,
��[UU[afmrz����zsmea[[;ABMNOZ[\\[NLB;;;;;;gntv����������~wqkgg�������� ����%)+-,*)"mea^_es�����������zm�����������������zÃÇÌÇÆ�z�q�n�k�n�r�z�z�z�z�z�z�z�z����������������������������������������������������������������������������������������� ��������������������������������	���%�%�#��	������������������ÓÔÕÓÑÇ�z�z�zÀÇÓÓÓÓÓÓÓÓÓ�����������������������������������������s�����������x�s�f�Z�M�D�C�D�C�D�X�f�s�����(�5�7�3�'����ܹϹ͹ù��ùϹ��Ź��������������ŹűŭŬŭŰŹŹŹŹŹŹ���������ĿοĿ��������������������������m�y�����������������y�`�T�D�<�>�C�L�X�m���5�A�N�T�^�a�^�N�A�5�(������������������� �����������������������������׻S�_�l�x�t�l�c�_�S�F�:�-�+�+�-�3�:�F�I�S������������������������������������{ŇőŔŠŦţŠŘŔňŇ�{�n�g�d�c�h�n�{���������������u�y�������������(�A�M�Z�b�c�]�Z�M�A�4�(���������������"�����������������#�#�/�8�<�=�?�G�<�/�#�"����#�#�#�#�#�T�`�a�m�p�n�m�`�^�T�K�G�?�@�G�J�T�T�T�T��"�.�7�5�7�0�.�"��	������������	���f�u�u�y�����s�Z�J�A�������,�4�E�f�����žʾ̾ʾǾ��������������������������������������������y�m�`�T�J�F�L�Y�`�y���ûлڻѻлû��������ûûûûûûûûûýĽнݽ�������ݽнĽ����������������ļ����ʼмҼͼ¼�������f�\�V�U�Y�e��������	��"�.�G�V�]�[�O�G�;�.�"�	���������(�5�A�M�X�Z�f�_�Z�N�5�����������)�@�O�h�v�{�z�p�^�B������������������������������������������������������$�0�=�B�<�2�$����������ƿ�����������$�{ǈǔǡǫǭǺǭǫǡǔǈǁ�{�t�x�{�{�{�{D�EEEEE%E(EEED�D�D�D�D�D�D�D�D�D��<�H�U�h�q�t�|ÇîîàÕ�n�H��� �%�0�<������	������������������ÓàìùýùùìàÓÏÊÓÓÓÓÓÓÓÓ�[�h�t�{�|�t�h�`�[�X�[�[�[�[�[�[�[�[�[�[������������������������������������������
��#�%�+�*�$�#��
�������������������A�N�Z�a�g�m�h�[�Z�N�G�A�@�8�5�2�2�5�=�A����������������������������������������²¿������������������¿»²­«²²²²���������������������������������������s�����������������s�r�j�f�Z�W�X�Z�f�sŠŭŹ��������������ŹŭŠŗŔŒŔŖŠŠ��"�#�/�/�/�"�!��	� ��	���������������������������������������������������!�-�3�<�?�:�6�!�����ֺʺպ����r�~�����������������������~�|�r�n�l�r�r���������ûлܻ�������ܻлƻ���������D{D�D�D�D�D�D�D�D�D�D�D{DqDoDlDkDoDoD{D{ A o T    w / ` ' 7 | 0 7  + , - b . ? F ! 8 l 4 3 V w < = 8 2 b L E " s R < A ( " +  L J <  D I E 9 M .  d  /  /      V     �  �  �  �  p  �  7  <  �  u    �    �  �  J  	  h  �  8  �  e  3    6  �  B  �  �  �  v  �  6  b  �  C      �  �  B  �  �  9  8    a  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  �    i  P  7    �  �  �  �  l  @    �  �  _  "  �  �  i  �  �  �  �  �  �  �  �  u  a  M  9  %    �  �  �  �  c  A  �  �  �  }  h  S  >  .  "    	   �   �   �   �   �   �   �   �   �  �  �  �  	�  
I  
�  �  �  $  <  I  '  �  Y  
�  	�  r  ;  O  �    �    J  v  �  �  �  �  f  C    �  �  &  �  �  �  �  �  �  �  �  	  
                       #  '  )  ,  /  o  �  �        �  �  �  �  z  M    �  U  �  �    F  c  B  U  f  u  {  z  u  g  S  .  �  �  �  �  �  a  -  �  �  �  l  q  p  h  N  +    �  �  �  �  �  �  �  �  �  E  �    �  K  L  L  F  =  2  %      �  �  �  �  �  �  �  �  �  �  �  F  :  -  !    �  �  �  �  �  l  E    �  �  �  i  5     �  �  V  �  �  �  �  �  �  �  o  8  �  �  v  *  �  )  �  L  o  	Y  	�  	�  	�  

  	�  	�  	�  	�  	o  	;  �  �  ;  �  �  9  B  �    \  �  �  �  �  �  �  �  �  i  9    �  n    �  g    �   �  W  T  N  A  -    �  �  �  z  I    �  �  x  N    �  �  -  ~  �  �  �  �  �  �  �  �  �  �  �  m  5  �  �  p  '  �  �  K  Q  U  W  V  O  D  <  &  	  �  �  �  n  ?    �  �  �  K  Z  V  Q  L  H  A  5  (         �  �  �  �  �  �  �  t  a  r  q  �  �  �  �  �  y  l  Y  D  -    �  �  ~  >  �  �   �  �  �  �  �  �  �  a  @    �  �  �  �  z  M    �  �  I   �  �  �  �  �  �  �  {  q  g  ^  V  M  @  2  #    �  �  �  r  �  �  �  �  �  �  �  �  �  �  �  q  ]  F  '    �  �  �  T  _  ^  ]  W  N  D  6  (      �  �  �  �  l  F     �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w    �    {  �          �  �  �  �  �  t  O  (  �  �  y    x  �  G    0  ;  =  :  6  0  $    �  �  �  C  �  �  L  �  A  �  �  �    |  z  y  w  s  n  e  Z  O  D  9  0  '  +  2  A  Z  t  �  �  �  �  e  I  *  
  �  �  �  �  t  T  9  -    �  ~    
f  g  ~  �  �  �  �  �  �  �  �  r  H  
�  
j  	�  m  �  u  �  �  �  y  p  e  X  G  2    �  �  �  \  #  �  �  �  |  I    �  �  �  �  �  r  [  F  ;     �  �  �  "  �    �  �  0  g  �  �  �  �  �  �  �  �  �  �  N    
�  
  	  �  �  r  ;  {     �  �  �  �  �  �  z  b  I  0    �  �  �  �  �  �  b  3  �  �        �  �  �  c    �  o    �  9  �  Z  �  ;  [  j  d  Z  M  -    �  �  a    �  �  M  �  �  R  �  �  9  �  	�  	�  	�  	{  	^  	;  	  �  �  c    �  )  �  ,  �     R  �  y  �  �  �  �  ^  8  @  %    �  �  �  K    �  �  B  �  9  _  n  �  �  �  �  �  �  y  g  U  A  ,    �  �  �  �  ~  W  1  �  �  �  �  �  ~  p  b  R  3      �  �  �  �  �  �  �  �  �  �  z  s  k  c  [  R  J  B  9  0  '      
  �  �  �  �  <  S  R  B  .    �  �  �  �  �  V  #  �  �  9  �  �  M  e    �  �  �  �  �  �  �  j  E    �  �  �  ]  '  �  n  �  t  |  r  a  K  1    �  �  �  i  4  �  �  �  b  /  �  �  }    �          �  �  �  �  d  .  �  �  �  ^  0      +  ;  �              �  �  �  �  �  @  �  �  m  4    �  �    �  �  �  �  �  `  =    �  �  �  |  T  1  �  �  0  �  j  E  +    �  �  �  _  2    �  �  {  k  Q  (  �  �  <  �    �  �  t  ]  B    �  �  �  �  k  =  
  �  �  H  �  h  �        �  �  �  �  �  o  J  $  "  C  \  5    �  �  �  h  =  �  �  �  �  z  V  ,  �  �  �  c    �  �    �  C  �  k  	  7  U  <    
�  
�  
�  
d  
%  	�  	�  	4  �  @  �  �  8  x  �  �        �  �  �  ^  0  	  �  �  �  ^  .     �  �  x  ]  4  �  R    
�  
�  
�  
n  
3  	�  	�  	  	7  �  q  �  �    f  �  �  �  �  �  u  <  �  �  g    �  e    �    
y  	�  	  b  �  >