CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�Q��R      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       >�u      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @Ej=p��
     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�|    max       @v[\(�     	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @N@           x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���          �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �D��   max       >]/      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B)��      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B)��      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?>�   max       C��      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?Na�   max       C��q      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          `      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P=�.      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��䎊q�   max       ?���?      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ě�   max       >�u      �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>޸Q�   max       @Ej=p��
     	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v[\(�     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @N@           x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�t�          �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D~   max         D~      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�6��C-   max       ?���?     p  S(               ,                     
         9   @         _   &   @            %      
   L   5               /                  ;      *         E   &   	            7   M         !   B      O   	O��N�hAO���O��O���O#��O&*<N�N-��O�7O}�N�K�N)AgO��P��PPSN�:QN�ՉPM7�O�R�PB�N[lN��O5O�O�ԇO1,{N���P,�2PVC>O�#�OwY7M�2�NN�(P'kkN=_�M�-8N��N7u�N�UPކO�sO���N�/[O+�@P-�,OB�MNr�N�ٺN�vN�<�O��O��N�"N�^�O�>O@��O}�OmCM����ě�����D���o��o��o;ě�<o<o<o<t�<49X<49X<T��<�o<�C�<��
<�1<�j<ě�<���<���<���<�/<�h<�<�=o=o=+=+=C�=C�=\)=\)=t�=0 �=8Q�=D��=ix�=�o=�o=��=�C�=�O�=�O�=�O�=�\)=�\)=���=��=�9X=Ƨ�=Ƨ�=��=�x�>�>I�>�ufgmt����������trmigf}|����������������}}:?AUan����}|ofa[UG:t�����������������|t�����������������������������������������������

��������������������������BBNO[ef[OBBBBBBBBBBB�������������������������")*)�����uz�������������zuuuu`]ht����th``````````}{�����������������}rkt�����  ������r�����
#<H<83#
�����../0<FHSUXURH<4/....)/6BOQROLJB6+)#���)5?@<<?>,������USSRJH</#
#/=HU��������������������GDBHU\]adaUHGGGGGGGG�����

�����������������������������")BO[cf^TOI@6)"(/7;?A@?;;/" �������������������������
/<SZ_XUH<,�OKKOPQUaz��������zaO������������������������""�������GBEHLUWVUHGGGGGGGGGG�������������������������A?83,#���??BOW[\ca[OB????????#!#'(#########������	������������MOO[fhihb[POMMMMMMMM)5>BCB65*)(gjq|�������������tjg��������������������4003APTaimid]YTHFA;4bbnn{������{nbbbbbbbzz{����������������z������������������������,''0<FIIIH<0,,,,,,,,����

���������#$/4<=<0//#80128;HMONJH>;888888�����#!0=DD@5)�yuonotz������������yaaaa\_ammwyz{zvmeaaa������� #$/;<FD?<;7/#  ��������� �������"#/<BAA<</$#"��������������������zz��������zzzzzzzzzÇËÓØÙÓÐÇÁ�z�n�i�a�[�_�a�n�zÀÇ�
���#�*�.�#��
�����������������
�
���ʾ��������׾ʾ����������������������Ŀѿݿ��ݿĿ����������������������������'�1�6�:�3�'�����ܹʹŹιܹ軑�����������ûƻɻ»��������������|�����)�5�.�6�B�C�F�B�=�6�)���������)�����������������������������������������ּ׼����ּ˼ͼռּּּּּּּּּּ�'�-�9�@�@�4�-�'�%��������������#�(�(�����������������������t�uĀāčĎđčĄā��t�q�s�j�r�t�t�t�t�L�Y�^�a�[�Y�L�K�F�J�L�L�L�L�L�L�L�L�L�L�������� �#����������������������뾌�������¾־��׾�����Z�@�<�C�A�a����������+�6�8�5�)�������������������������������������������������������àæéìóøïìàÙÓÇÁ�~ÇÊÓØàà�y�������ʿǿ������y�`�T�;�*�/��"�.�7�y�m�`�T�G�;�:�9�>�F�Q�`�m�y�����������w�m�/�T�a�m�|��������m�T��	�������������/àìù����ùìâàÖßßàààààààà�����������������r�q�r�r�x�����������������������������������������������~���������˺ú������������r�e�\�W�e�r�~�H�T�a�n�x�z���z�w�m�a�T�H�<�;�3�9�;�E�H�������������������޹�������/�;�H�Q�Z�\�Z�T�;�/������������������(�5�g�����������������Z�N�2�(���������������������������{�g�Z�S�Z�Z�e�r���E�N�R�O�L�A�7�5�(���
������(�5�EEPE\EiEkEiE_E\EPECEOEPEPEPEPEPEPEPEPEPEPàáìùúùöìàÓÒÌÓØàààààà�����
�#�A�K�N�K�0�#���������ľĿĶĿ���
����#�$�#��
����
�
�
�
�
�
�
�
�~�z�~���������������������~�~�~�~�~�~�~�H�U�Z�U�I�T�H�<�/�/�#��#�/�<�C�H�H�H�H�4�?�@�L�@�;�4�1�'�!�'�+�4�4�4�4�4�4�4�4�(�,�4�6�7�6�4�,�(�&������� �"�(�(�s�����������������������g�X�K�E�W�g�s�`�T�G�?�;�9�>�G�H�T�`�m�y�{�{�y�s�m�a�`�
��#�0�I�U�[�[�I�#��
���������������
��������������������������������нݽ��ݽ�ӽĽ��������������������Ľ��
�#�<�H�U�\�]�X�I�<������������������
�!�-�:�F�I�O�P�F�E�:�-�!���������!�f�s�������t�s�l�f�c�c�f�f�f�f�f�f�f�f����������������������������������������²½¿��¿¶²¯¦¥¦®²²²²²²²²�tāčĚĦĪıĦĚčā�z�t�q�t�t�t�t�t�t�����)�6�B�G�B�6��������������������:�F�S�l�x�������������x�l�S�F�0�%� �)�:��������������������������ŽŹŰŹ�������I�V�b�c�o�b�W�V�I�=�2�.�$�"�$�0�1�=�A�I�����������
�
��
�
��������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D~D{DwDtDxD{D{�����*�6�8�>�6�*������������������ʼ���� ����������ּʼ��������������������ݽ׽ݽ����������������� * 7 2 H - 2 0 8 b C N p a = > 6 . ? J 9 H V I N B ( L R - > * b 7 = T l \ - ] k 6 R p @  & $ > \ P +  ^ f R  5 3 y    I    �  �  �  g  k  <  l  U  ,  �  n  �  d  �  �  '  �  K  �  v  �  �  �  |  �  e  �  >  �    e  �  w  -  �  =  �  i  5  �  �  �  �  �  z  �  I  �  �  �  �    =  �  4  �  5�D����o<���<�1=0 �<�o<���<D��<�o<��
<��
<��
<�t�=��=��=��
=,1=C�=��=}�=�Q�<�=�P=P�`=�+=H�9=�w=�"�=�1=]/=m�h='�='�=��T=�w=#�
=}�=ix�=]/=�h=��w=�
==��-=���>I�=�"�=��w=�{=���=�j>C�>(��=�/=�
=>1'>7K�> Ĝ>]/>!��B
LB iB<�B�B��B"��B��B?�B�1B!KiBB�B�BScB-lBwHB��B��B��B3�BҲB�aB#��B�RB�6A��B {B%�Bj�B��B�B�dB!�lBX�B�lB%B�4B<�B�{B
�
B!]�A�?�B(�IB)��B+�B��B&$BĩB�A���BQB�xA�p�BtJB/�Bh�B�B�B��B
?�B
�FB@qBF�B��B"�B�B@B<�B!I8B��B9�B<�B��B�3BI�B�B�PB�B>gB�KB�WB#��B;VB� A���B��B9BGSBEyB7�BBOB"?BrB�vB%;�B�#B?�B�$B@B!?$A��B(�@B)��BAB��B&7�B��B?�A���BVBB�A��B?�B@�B�zB2�B�B��A�jA��ARvAu��?T%0@�6�A��A�t/A��@�F�A�F�Aݏl?�5%A�?hAI�A��A�	FA�QAl;�Ai��A�|�A̖B@�n	A�5R@�\A��?>�A��LA��A��xA��hC��A˽�A���A��@�Aò7@ι�A75KA�,�Ah
A�%G@U�(A&�1A�R@q�ACmxA�m+A��iAަ�AӲs@�ςA�v�B"�A���C���A��A~�A.A�iOA�rlAS�At�~?Ox�@��;AՔkA���A d@ŝ�A�DA݀t?ւJA�d4AC^�A��A�z0A�Ak@�Ak�A�9�A̽S@��^A���@>`A�o�?Na�A�k�A�{gA�yZA���C��qA�5�A�%A��N@�A�
@��yA8�eA���Ah�|A��@S��A%V\A��y@s��ADA���A���Aރ�A�r�@���A��)BT�A�xlC�ʜA���A�]A/_                -            	         
         :   @         `   '   A            &      
   L   5               /                  <      *      	   E   '   
            7   N      	   !   B      P   
         %                                    ;   '         1      /            #         -   +               +                  1               '                  #   !                                                                  +                  #                        '               #                  '               %                  !                        O��N�*kO��O��O{�]N�O&*<N�N-��O�7O}�N+�4N)AgO��P4��O��iN�<�N�ՉOǁ&O�R�O�f^N[lN0K0O5O�O1P0O-NA��O���P=�.O�OZ�M�2�NN�(O�C�N=_�M�-8N,-�N7u�N9�P�EO�sO���N�/[O+�@P�	O.'�Nr�N�ٺN�vN�<�O��rO�wN�"N�^�N��O@��O}�OId�M���  �  �  �  �  �  N      G      >  i  �    d  �  }  
z    #  �    �  V  �  L  f  ^  �  o  1    �  p  �    ,  �  	a  H  �  �  �  
Q  �  �  �  �  �  
�  
i  #  �  �  �  	a  �  ��ě��u��`B��o<#�
<o;ě�<o<o<o<t�<D��<49X<T��<�=#�
<ě�<�1=ix�<ě�=0 �<���<�h<�/=,1=o=o=ix�=�P=C�=\)=C�=C�=H�9=\)=t�=<j=8Q�=L��=�%=�o=�C�=��=�C�=��-=�t�=�O�=�\)=�\)=���=� �=�E�=Ƨ�=Ƨ�=�"�=�x�>�>z�>�ufgmt����������trmigf��������������������F?DFMUanz��z~yxjaULF�����������������������������������������������������������������

��������������������������BBNO[ef[OBBBBBBBBBBB�������������������������")*)�������������������������`]ht����th``````````}{�����������������}�������������������������
$('#
����7004<CHQRHE<77777777)/6BOQROLJB6+)#����),22.)"����USSRJH</#
#/=HU��������������������GDBHU\]adaUHGGGGGGGG������	

������������������������������$)6BOQSNF?6/)'"//;=?><;6/$"��������������������#/<MQTSMH</
NQRUaz���������znaUN��������������������������� !����GBEHLUWVUHGGGGGGGGGG������������������������),-*+%����??BOW[\ca[OB????????#!#'(#########������������������MOO[fhihb[POMMMMMMMM)5;>50)$njls��������������tn��������������������6215=ETaegkha[WTHD;6bbnn{������{nbbbbbbbzz{����������������z�����������������������,''0<FIIIH<0,,,,,,,,����

���������#$/4<=<0//#80128;HMONJH>;888888����� +9BB?5)��yupnotz������������yaaaa\_ammwyz{zvmeaaa������� !#*/4<A?<;84/##  ��������� �������"#/<BAA<</$#"��������������������zz��������zzzzzzzzzÇËÓØÙÓÐÇÁ�z�n�i�a�[�_�a�n�zÀÇ���#�'�*�#��
���������
������������ʾ����������׾ʾ������������������Ŀѿܿ�ݿҿĿ���������������������������'�0�3�*�'������ֹܹй׹ܹ�������������������������������������������)�5�.�6�B�C�F�B�=�6�)���������)�����������������������������������������ּ׼����ּ˼ͼռּּּּּּּּּּ�'�-�9�@�@�4�-�'�%��������������#�(�(�����������������������tāččĐčā�t�r�t�t�t�t�t�t�t�t�t�t�t�L�Y�^�a�[�Y�L�K�F�J�L�L�L�L�L�L�L�L�L�L�������� �#����������������������뾌�������ʾھھξ�����f�Z�T�Q�U�d�o�������������$�(�'��������������������������������������������������������àæéìóøïìàÙÓÇÁ�~ÇÊÓØàà�`�m�y�������������m�`�T�L�G�?�>�C�G�S�`�m�`�T�G�;�:�9�>�F�Q�`�m�y�����������w�m�T�a�k�y�|�v�m�a�T�H�;��	��������/�Tàìù����ùìâàÖßßàààààààà�����������������x�x�~���������������������������������������������������������������������~�r�g�e�d�e�r�~�����H�T�a�l�m�v�z�z�s�m�a�T�H�F�;�6�;�<�H�H������������������������"�/�;�H�M�N�K�B�/�"��	�����������	��"�5�g�����������������Z�N�7�(������5�����������������������s�g�[�Z�W�[�f�s����(�5�A�N�N�N�K�A�5�(������	���EPE\EiEkEiE_E\EPECEOEPEPEPEPEPEPEPEPEPEPàáìùúùöìàÓÒÌÓØàààààà���
�#�4�=�>�9�0�#��
�������������������
����#�$�#��
����
�
�
�
�
�
�
�
�~�z�~���������������������~�~�~�~�~�~�~�H�U�W�U�H�F�H�Q�H�<�6�/�#�#�#�/�<�D�H�H�4�?�@�L�@�;�4�1�'�!�'�+�4�4�4�4�4�4�4�4�(�(�4�6�6�4�(� ����(�(�(�(�(�(�(�(�(�g�s�������������������������g�Z�O�K�]�g�`�T�G�?�;�9�>�G�H�T�`�m�y�{�{�y�s�m�a�`�
��#�0�<�I�U�R�I�<�#���������������
��������������������������������нݽ��ݽ�ӽĽ��������������������Ľ���#�<�H�R�Z�Z�T�H�<��
����������������!�-�:�F�M�N�F�A�:�-�!��	�������!�f�s�������t�s�l�f�c�c�f�f�f�f�f�f�f�f����������������������������������������²½¿��¿¶²¯¦¥¦®²²²²²²²²�tāčĚĦĪıĦĚčā�z�t�q�t�t�t�t�t�t�����)�.�6�>�5����������������������:�F�S�l�x�����������x�l�_�S�F�1�&�!�*�:��������������������������ŽŹŰŹ�������I�V�b�c�o�b�W�V�I�=�2�.�$�"�$�0�1�=�A�I����������
��
�����������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D~D{DwDtDxD{D{�����*�6�8�>�6�*������������������ʼּ������������ּͼʼ������������������ݽ׽ݽ����������������� * : 5 B . ? 0 8 b C N : a = 7  . ? ( 9 : V Q N B " D : $ = % b 7 0 T l w - A b 6 I p @  ' $ > \ P -  ^ f S  5 + y    I  �  &  @  �  M  k  <  l  U  ,  ;  n  �  D  &  �  '  �  K  #  v  G  �  �  C  r  �  3    �    e  �  w  -  �  =  X  �  5  /  �  �  z  j  z  �  I  �  �  �  �    �  �  4  �  5  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  D~  �  �  �  �  �  z  Y  7    �  �  �  H    �  l    �  A  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  d  R  :    d  �  u  �  �  �  �  �  �  �  a    �  g  q  �  �  I  �  ]  �  :  d  �  �  �  y  t  i  _  U  @     �  �  �  X  %  �  �  �  :    H  q  �  �  �  �  �  t  S  "  �  �  Y    �    P  -   �      $  <  I  K  K  K  K  L  N  O  C  2      �  �  �  v                 �  �  �  �  �  w  K    �  �  v    �    �  �  �  �  �  �  z  q  i  Z  D  .      �  �  �  p  J  G  4  !    �  �  �  �  �  �  �  q  ^  P  A  C  K  U  a  n      �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  (  �  I          
    �  �  �  �  �  �  �  �  s  \  @      �  �  �  7  -      �  �  �  �  u  Y  >  %    �  �  �  �  �  i  i  h  e  _  Y  P  C  7  (    
  �  �  �  �  �  �  �  g  �  �  �  �  r  ^  H  +    �  �    g  H  2    �  �  {  �  �  �  �  �        �  �  �  L    �  �  8  �  �  �  �  �  8  �  �    5  N  `  d  V  ;    �  �  R  �  �    d  �  �  w  �  �  �  �  �  �  �  �  �  x  \  <    �  �  �  \  3    }  u  l  b  V  I  9  &    �  �  �  �      �  �  �  �  c  �  �  	5  	�  
  
X  
u  
w  
]  
-  	�  	�  	  �    �  �  �  �  �    �  �  �  �  �  �  �  r  Q    �  w    �  c  �  �  �  u  ;  {  �  �    #      �  �  �  �  ;  �  U  �  ;  �  H  �  �  �  �  �  �  �  m  X  D  /      �  �  �  �  }  Y  5    �  �  �  �      �  �  �  �  �  �  �  �  z  i  W  D  0    �  �  �  �  �  �  n  G    �  �  �  \  "  �  �  f  7    �    0  @  F  E  I  S  V  O  <    �  �  ~  �  �  Z  ,  =  �  �  �  �  �  �  �  �  �  �  �  n  V  -  �  �  >  �  H  �  6    "  *  9  I  L  K  C  8  &    �  �  �  �  �  j  I  $     �  �    >  N  X  e  b  S  ;    �  �  ,  �  /  �  �  �  �  9  R  \  N  7    �  �  d    �  <  �    �  *  �  q  �  �  �  �  �  ~  b  ?    �  �  �  �  �  `  :    �  �  �  t  H  a  m  l  e  Z  J  3    �  �  �  �  E    �  b    �  I  	  1  !    �  �  �  �  �  �  n  K  $  �  �  h    �  �  2  �      �  �  �  �  �  �  �  }  y  t  p  l  p    �  �  �  
  j  �  �  �  �  �  �  �  �  e  -  �  �  Z    �    �  ~  b  p  m  j  g  c  ^  T  J  ?  5  *        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  i  Y  I  8  '      �  �  �    q  �  �  �  �  �  >  3  (      �  �  �  �  �  ,  %      �  �  �  �  �  o  D    �  �  �  N    �  �  v  �  �  �  �  �  �  �  �  �  �  �  �  �    d  �  "  w  �     �  	O  	`  	Q  	B  	1  	  �  �  W    �  z    x  �    #  e  �  H  ,    �  �  �  �  z  X  4    �  �  �  P    �  �  U   �  [  �  �  �  �  �  l  I    �  �  c    �  X  �  m  �  p  �  �  �  �  v  W  8    �  �  �  �  c  5    �  �  i  1  �  !  �  �  i  N  2      �  �  �  �  �  �  �  �  �  �  �  �  �  
/  
M  
Q  
K  
;  
  	�  	�  	c  	  �    j  �  5  �  �    $  �  �  �  �  �  �  �  �  �  u  5  �  �  U     �  K  �  �  �  �  �  �  �  �  �  �  t  d  S  =  #    �  �  �  �  �  �  5  �  �  �  �  �  �  �  h  R  @  2  "    �  �  �  �  �  �  �  �  �        �  �  �  �  x  T  /  
  �  �  �  k  B    �  �  �  �  �  �  t  c  S  C  ,    �  �  �  y  N  "  �  �  `  �  
�  
�  
�  
�  
l  
6  	�  	�  	o  	  �  U  �  L  �  X  �  �  �  ^  
h  
h  
[  
I  
?  
/  
  	�  	�  	�  	\  	  �  \  �  R  �  �  �  j  #  �  �  �  ]  )  �  �  �  �  s  R  0    �  �  O  
  �  @  �  o  \  F  .    �  �  �  �  �    g  A    �  �  �  �  i    �  �  �  �  �  �  ^  $  �  �  0  �  0  �  �  ^  �  ,  �  �  �    L    �  �  1  �  l  �  e  �  
�  	�  �  �  T    �  	a  	,  �  �  z  F  �  �  �  K  
  �  n    �  2  �  &  t  -  �  �  �  �  �  �  ^  (  �  �  &  �  /  �  
�  	�  �  �  H  �  �  �  �  �  g  `  _  \  X  Q  G  7  !    �  �  �  �  �  �