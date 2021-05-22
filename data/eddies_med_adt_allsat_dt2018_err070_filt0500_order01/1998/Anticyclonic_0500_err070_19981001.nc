CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?öE����      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�   max       PύR      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       =���      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=q   max       @Fz�G�     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v~�G�{     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @3         max       @N�           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�D�          �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��1   max       >���      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B/��      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��}   max       B/��      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��_   max       C��B      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�U�   max       C��$      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          E      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�   max       P�=�      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����)_   max       ?�0��(�      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ě�   max       =���      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=q   max       @Fz�G�     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v}�Q�     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @N�           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�f�          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�$tS��N   max       ?�0��(�     �  M�                  !   X               R   h            �                   F                  .      
   (      <   �   �                     1         !   8   2      :      9   �N���N���N]��N�cqN~XVO���O�tO�vnOW<*O7�N�ۘP�`�PύRN���N���N�t�PUGOe�O5COM��O�/NU�O�/>O�b�Nձ�N�0�O��wO�ޑPe�OR.eO1H�OɲKO��O��PN�MP+��N��O$�N�a�OK��O�>N��O��Ou;PN�Oɿ�O��BO�6�N���O��KN�W�O�c�O���ě��e`B�o��o��o%   :�o;ě�<49X<T��<T��<�C�<�t�<�1<�1<ě�<ě�<���<�`B=C�=C�=C�=\)=�w=#�
='�='�=,1=,1=0 �=49X=8Q�=@�=@�=D��=P�`=T��=T��=Y�=Y�=ix�=u=y�#=y�#=�C�=�O�=���=���=���=�1=�1=�v�=�����������������������??@BOQW[a][OGB??????�����������������������������������������������������������������������������ifbdhnz�����������zi����)6<??:)������������������������.*',/<HLUa_WUH</....vv������������vvvvvv�����
#+-6KUJ#��������Nt��t_WQB6)���SU[\gtx���tg[SSSSSSmnoy{������������{nm"#(//<?EC<</#����������������������������������������������

�����(')),5BNR[gqnjgXNB5(�|����������������������	

��������]em������������zmgc]����
#<JIC@</#����������������������*/67762*(*****INUU[gt�������tg^[OIhjiinz�����������zmh��)5;N����gNB5�������� !����	)5:=<:<5-)'	6BO[_dec[OB6)�������� �����������
/8=?<2#	 �������������������%"BNgt���������gB5%��������������������"+/;DHKT]bbaTH;/"������

�������#/<NUXZUH</#" �
#0<>HA<50#
  3456=BLOOZTOIB>63333
)6;CDHB61)#��������������������*6@;6*���)6BEB6������"# �������������� # ����
)*/1,)!����������� ��������� #)*)'���������������

�����������

����������������������������������������������Ž������������������y�v�o�y�|�������������;�G�I�K�I�G�;�1�.�)�.�6�;�;�;�;�;�;�;�;�'�,�3�8�?�@�C�@�3�'����
�����'�'�������ûлۻл̻û�����������������������(�5�A�N�\�p�{���s�g�N�A�<�8�5����E�E�FF$F1F;FEFIFGFAF1E�E�E�E�E�E�E�E�E����	��"������������������������������'�*�,�/�4�4�'�������������	������������
����������������������/�<�=�E�B�<�5�/�%�#���#�&�/�/�/�/�/�/����-�:�E�S�T�F�!������׺ɺ����Ϻݺ�Ƨ��������������ƧƎ�h�F��
��*�\�tƧ�������������������������������������������������!�+�-�5�-�!�������������D�D�D�EEEEEED�D�D�D�D�D�D�D�D�D�D�(�4�A�M�S�P�A�1����ν������y�����ݾ�(ĚĦĮĳĵĵĳĲĬĦĚčĈĈćĊčďĚĚ�(�3�5�=�A�I�P�Y�X�P�N�A�5�(�'�%�%�$�"�(����(�4�@�E�B�A�=�<�4�(����
�	���)�6�9�6�3�2�)�!����������� ����)�.�;�G�K�J�G�;�.�"����"�)�.�.�.�.�.�.�������������������ŵŭūţŮŹ���߾(�A�M�f�s���������x�f�M�A�(������(�)�6�<�B�L�Z�Z�O�B�;�7�6�3�)�����&�)���������y�v�m�e�g�m�y�������������������ݿ������� �� ���ݿĿ����������¿ѿ������������������������������������������;�P�P�H�;�#�	������������������������N�[�g�k�t�}�t�[�N�B�=�5�*�5�B�G�N���������Ŀ����������������y�v�y�����������׾����������׾ʾ���������������������	���������׾ʾ����������ʾھ�G�`�y�����������y�m�`�S�G�A�?�;�.�"�.�Gì������������������ìàÓ�z�a�]�X�nÓì�������#�1�=�=�6�)�������������������ÆÃÂÅÇÓßàçìöìæàÓÒÆÆÆÆ�#�0�6�<�?�F�I�J�I�F�<�0�#�#������#�;�H�T�W�_�a�`�W�T�H�B�;�7�6�4�7�;�;�;�;�/�5�:�7�-�"���	�������������	��"�)�/���������	��������߼�����r�����������������������t�r�i�r�r�r�r��!�-�<�I�P�K�F�:�!������ܺ������y�������������������������y�l�b�e�l�o�y�S�^�`�f�g�`�S�P�M�Q�S�S�S�S�S�S�S�S�S�S�Y�r�~���������ºĺ������q�]�Y�L�@�8�5�Y���������������������������������������#�0�<�I�U�\�m�s�m�W�I�0��������#ĳĿ������������������������ĿĴĳĬĳĳ���ܼ��"�%�$�������лû�������������'�.�4�:�<�4�*�'�"����������EuE�E�E�E�E�E�E�E�E�E�E�E�E�EuErEoElEjEuD�D�D�D�D�D�D�D�D�D�D�D�D{DuDtD{D�D�D�D� z N C 2 P < J 9 > 2 V B K 4 ` K O O o @ h o 6 & _ < , D i : 4 * x 4 - H m Y H E M g ' I _ a ; ( U 9 3 1 '      �  g    �  ]  N    �  %  �  Z    �  F  �  �  t  �  �  p  �  0    �  �  Z  �  1  �  �  �  �  �  �  8  �  �  ?  �  t  �  y    \  #  F  �  5  "  �     A��1��`B�ě�<u;o=o=�E�<�j=C�=C�<���=���=��#<���=C�=L��>!��=m�h=H�9=@�=8Q�=�P=�
==�+=y�#=@�=�7L=�+=�-=�7L=Y�=�{=�o=��>B�\>7K�=�t�=�o=y�#=�\)=�t�=�7L=�;d=�E�=�t�=��`>+>+=��`>hs=�^5>�u>���B��B�NBˢB!$B#XB�B�WB��B!:B�RB`bB"�]B�OB	U.B)mBM'B"8B�IB��B!�B��B��B �B��B��B/��B	�B �EBKBN`B�FB��B	%B>�B��B	!DB"��A���B(�B�jB%-�BITBLB,e�B/��B��B�<B#'B&BӎB^{B��B��B��B��B��B!;�B#O�B��B�B��B!�B=�B��B"EB�WB	N�B)<�BB9B"E~B��B?B��B��BÐB��B��B��B/��B
:RB @�B��B?�BI�B/SB��B5�B�HB�KB"@*A��}BЅB��B%=�B��BȕB,<�B/��B��B��B;�BCJB��B��BA�B��A���A�Ac�_?��_@�V�A���C��BA�T�@���A���A�@X�wB<lAG��@b�C�J A.��Aߗ9A��A6��A�g�Aad�A�W�A= �A��AnXA|C�A���A��A�ԶAsf�AP��AT�zAjc�A�t�A�F�A��A��A���A���A��@��@e��AE�A�j@V@A�k�A�(�A�!W@�5�@ȈC��C�ߊA�1�AHAc/�?�U�@�ĒA��SC��$A���@�cA��A@TB�AHho@\NC�HA.�)A߉�A��cA6AԤ3A` �A��=A<��A�Y9Am!A|�oA��uA���A���As�sAP�}AT�jAj�xÂ�AӉ�A˄�A�PA���A�x)A�@���@cp4A*XAM@	mA��A��A�@���@ȕ�C��C��                  !   Y               S   i            �   !               G                  .      
   )      <   �   �         	            1         "   9   2      ;      9   �                     !               9   I            3                  #   #               7         !      #   /   )                              %            !                                                E            !                  #   #               3                     #                              %            !         N���N���N]��N��fN~XVN���O�vnOi�OD�N�PN}�OB�P�=�N���N���N�t�O�2Oe�O5CO�&NɼNU�O�/>O�b�N]�aN�0�O6��O�ޑP]��O*��Oi�O��)O�lObhEO�O�!tNe��O$�N�a�OK��O�>N��Oh�#OE��N�O�n�O��O���N���O��N�W�O]�+O��  �  �  /  �  m    �  h  �  7  =  �  y    �  
T  -  �  Y    �  �  
b  �    F  �  �  R  }    �  �  	�  �    �  �  �    �  �  �  �  �    	�  	  �  	�  ?  �  ޼ě��e`B�o%   ��o<u<�1;�`B<D��<�t�<u=�C�<���<�1<�1<ě�=��-<���<�`B=t�=t�=C�=\)=�w=<j='�=H�9=,1=0 �=<j=<j=Y�=D��=u=�h=��-=aG�=T��=Y�=Y�=ix�=u=�t�=��=�C�=�\)=��-=� �=���=�9X=�1=ȴ9=�����������������������??@BOQW[a][OGB??????�����������������������������������������������������������������������������lhjpz�����������{wpl��)-6;>>96) �����������������������-./7<AHTUZUPH<3/----z�������������zzzzzz���������� ����������Ng|bSIB5)�����SU[\gtx���tg[SSSSSSmnoy{������������{nm"#(//<?EC<</#������������������������������������������������

�����,,05@BDN[gkjg[SNB;5,������������������������	

��������]em������������zmgc]����
#<JIC@</#����������������������*/67762*(*****YXcgs���������tlgd[Yhjiinz�����������zmh��)5C[j����gNB5����������
)57;9654))6BOW^`_[OB6)�������  �����������
#&/38:4/$#
����������������������2/7CN[gt������tg[B52��������������������"+/;DHKT]bbaTH;/"������

�������#/<NUXZUH</#" �
#0<>HA<50#
  3456=BLOOZTOIB>63333
)6;=@A=6)��������������������*6@;6*���)6BEA6)
��������!" �����������"�����
)*/1,)!��������������������� #)*)'�������������

������������

����������������������������������������������Ž������������������y�v�o�y�|�������������;�G�I�K�I�G�;�1�.�)�.�6�;�;�;�;�;�;�;�;�'�'�3�5�<�?�3�'������&�'�'�'�'�'�'�������ûлۻл̻û����������������������5�7�A�N�Z�\�e�]�Z�N�A�9�5�(�'�!�(�2�5�5E�FF$F1F;F>F=F1F$FE�E�E�E�E�E�E�E�E�E����	�����	�����������������������������'�+�.�3�'�%�������������������������������������������������/�9�<�C�@�<�3�/�)�#���#�*�/�/�/�/�/�/������$�*�(�!���������ۺ޺���Ƴ������ ����ƳƎ�h�X�+���,�\�zƎƧƳ�������������������������������������������������!�+�-�5�-�!�������������D�D�D�EEEEEED�D�D�D�D�D�D�D�D�D�D�ݽ��(�3�6�/�%�������ݽ̽������Ľ�ĚĦĮĳĵĵĳĲĬĦĚčĈĈćĊčďĚĚ�(�3�5�=�A�I�P�Y�X�P�N�A�5�(�'�%�%�$�"�(��(�4�<�A�A�A�?�9�8�4�(���������)�6�6�6�+�/�)������� �����#�)�.�;�G�K�J�G�;�.�"����"�)�.�.�.�.�.�.�������������������ŵŭūţŮŹ���߾(�A�M�f�s���������x�f�M�A�(������(�6�8�B�D�O�R�O�D�B�A�<�6�*�)�'� �)�0�6�6���������y�v�m�e�g�m�y�������������������ѿݿ�����������ݿѿĿ����������ĿϿ����������������������������������������	�;�P�P�G�/������������������������	�[�g�t�z�t�f�[�N�D�B�5�5�?�B�N�S�[���������������������������|�������������ʾ׾�������׾ʾ����������������ʾ���	���	� ���׾ʾ����������ʾ׾��`�m�y�����������~�y�m�`�T�N�G�G�F�H�S�`àìù��������������ùìàÓÇÄÁÆÓà�������)�2�7�6�1�)������������������ÇÓàìôìäàÓÇÃÆÇÇÇÇÇÇÇÇ�#�0�6�<�?�F�I�J�I�F�<�0�#�#������#�;�H�T�W�_�a�`�W�T�H�B�;�7�6�4�7�;�;�;�;�/�5�:�7�-�"���	�������������	��"�)�/���������	��������߼�����r�����������������������t�r�i�r�r�r�r�����!�-�3�:�B�G�B�:�-�����������y�������������������������y�r�l�i�k�s�y�S�^�`�f�g�`�S�P�M�Q�S�S�S�S�S�S�S�S�S�S�Y�r�~�����������º������r�g�]�Y�L�>�;�Y�������������� �������������������������#�0�<�I�Z�j�q�k�T�I�0�����
����#ĳĿ������������������������ĿĴĳĬĳĳ�ܻ��� �#�!�����лû������������ܼ�'�.�4�:�<�4�*�'�"����������E�E�E�E�E�E�E�E�E�E�E�E�E�EvEsEqEpEuE~E�D�D�D�D�D�D�D�D�D�D�D�D�D{DuDtD{D�D�D�D� z N C C P ; M , ; % L  J 4 ` K < O o ? e o 6 & l < & D h 5 / & w   9 X Y H E M g % K _ \ : & U < 3 + '      �  g  �  �    v  �  �  �  �  �  }  �  F  �  �  t  �  W  )  �  0    �  �  �  �  �  �  '  ,  �  �  �  !  p  �  ?  �  t  �  �  �  \  �  :  ~  5  �  �  �  A  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  Y  E  1    	  �  �  �  �  �  �  w  h  Z  J  ;  *    �  �  �  �  c  8    /  0  2  3  4  6  7  8  :  ;  9  4  /  +  &  !          �  �  �  �  �  �  �  �  �  �  �  �  `  ;    �  P  �  A  �  m  h  c  ^  X  R  L  E  =  5  -  $      �  �  �  �  J    G  U  l  �  �  �  �          �  �  �  v  ,  �  �  0  �  �  �  �  �  �  �  �  �  5  
�  
  	K  �  �  D  �  �  k  x  D  .  b  b  X  L  >  .      �  �  �  �  {  ]  E  .    �  �  �  �  �  �  �  �  k  W  x  ~  i  J  &  �  �  �  �  �  �  c      &  0  5  5  .  !    �  �  �  u  @    �  {  	  T      .  :  =  <  8  1  &    �  �  �  y  9  �  �  j    �  ~  3  L  q  �  �    ^  �  �  �  �  �  �  �  �  9  �    6  o  M  w  v  i  T  U  `  _  a  =    �  �  �  ]  �    �  �  @                �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  Z  E  3  )    �  �  �  �  V  �  �  �  
T  
'  	�  	�  	�  	N  	  �  �  R    �  n  �  5  e  �  �  �  �  
A  
�    q  �  �    '  ,    �  �  O  
�  
U  	�  [    �  �  �  �  �  �  �  �  h  J  %  �  �  �  t  :  �  �  \  �  B  �  Y  7    �  �  �  �  �  w  I    �  �  �  c  <    �  �  �              �  �  �  �  �  �  �  �  c  <    �  �  j  �  �  �  �  �  �  x  l  q  j  N  ,    �  �  �  �  g  �  v  �  �  �  �  �  �  �  �  �  }  q  e  Y  K  :  *       �   �  
b  
<  
N  
W  
X  
R  
B  
&  	�  	�  	�  	�  	^  �  h  �  5  u  y  �  �  �  �  �  o  T  9        �  �  �  P  4  $  �  �  \  �  
  x  �  �  �            �  �  v  Z  >     �  �  �  �  �  F  4  #       �  �  �  �  �  �  �  �  }  f  P  8       �  �  �  �  �  �  �  �  �  �  �  �  �  s  G    �  �  Z  (  �  �  �    _  :    �  �  �  [  1    �  �  y  /  �  '  �      A     �  �  �  [    �  �  @  �  �  �  [  �  J  �    �  t  v  {  {  n  ]  H  /    �  �  �  f  3  �  �  {    �  �  �    	        	    �  �  �  �  �  �  n  F    �  �  �  j  �  �  �  �  �  �  �  �  t  S  )  �  �  ^  �  o  �      �  �  �  �  �  �  {  `  C  #    �  �  �    �  !  �  R  �  	O  	y  	�  	�  	�  	�  	�  	�  	w  	>  �  �  =  �  ?  �  �       K  	�  
�  ~  :  �  {  �  [  �  �  �  d    �    /    	]  9    �  �  �        �  �  �  _  �  �  �  j  �  "  
@  	
  k  "  n  c  �  �  x  ]  :    �  �  T    �  �  N  �      �    �  �  t  f  ^  ]  l  p  n  g  ]  Q  <    �  �  x  >  �  N  �  �  �  �  �  �  �  �  v  h  X  D  /    �  �  �  X   �   q          �  �  �  �  �  �  }  Q  "  �  �  �  e  >  �  �  �  �  �  �  �  p  \  J  8  %    �  �  �  y  T  7  G  e  D  �  o  Y  E  5  %          �  �  �  �  �  n  O  [  x  �  _  �  �  �  �  �  �  k  L  '  �  �  ]  �  N  �  C  �  �  �  �  �  �  �  �  �  �  �  u  c  T  B  &    �  �  :  �  X  �  �  �  �  �  {  j  X  F  4  "    �  �  �  �  �  �  �          �        �  �  �  s  4  �  �  o  F  %  �  �  l  �  	�  	�  	�  	�  	�  	�  	�  	p  	F  	  �  �  /  �  g  �  -  �  �    	  	  	  �  �  �  �  y  D    �  f    �  B  �  '  s  �  �  �  \  -  �  �  �  �  �  �  n  @    �  �  H  �  �  �  �   �  	�  	�  	�  	{  	`  	<  	  �  �  R  �  �  +  �  ;  �  4  �  �  �  ?  *       �  �  �  �  �  g  M  4    �  �  �  �  �  v  b  �  �  �  �  �  q  1  
�  
�  
5  	�  	b  �    �  <  L     X  c  �  �  �  �  U  �  �  +  �  3  �  
  N  q  k  �  e  G  �  f