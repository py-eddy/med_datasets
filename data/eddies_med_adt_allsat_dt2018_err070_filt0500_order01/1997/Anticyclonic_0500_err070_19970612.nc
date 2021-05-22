CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?��x���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�8   max       Q P�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��h   max       =�l�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=q   max       @E\(��     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vt�\)     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P�           x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Α        max       @�X           �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���
   max       >k�      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B,n~      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�_�   max       B,@�      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��   max       C�~      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?4T   max       C�w�      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          U      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          E      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�8   max       P���      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��!�R�=   max       ?�3��ߥ      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��h   max       =�l�      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=q   max       @E�z�G�     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(�    max       @vt�\)     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @M�           x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�_        max       @�J�          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��1&�y   max       ?�0U2a|     �  T   
   	   $                  Y   	   m               .      9      
   	   /                              #   z         *      %               !      	   	   �   r   �               ;      ,      (   &   
Nx//N��O�LOOv]PRtN��zPy�OOw7Q P�O��PT�zN.�NP3O��OA��O�'O��O�;Nk�N� �N�f�PίN���N��#N���O��N��?O1O�N���OVP�O��P�NOë4O��!N�=aO�ƕN*i1OdM�8NռgN��7NSaO��@O�w�N�n�N�/�Pgg�PD�\P��OT�kNbv�O&2O���O��OD��O���O!��O_8aO4اN����h��1�49X�o;o;o;D��;ě�<t�<t�<49X<D��<T��<e`B<�o<�o<�C�<�C�<�C�<�t�<���<���<�1<�1<�9X<�9X<�j<���<���<�/<��<��=o=o=t�=t�=t�=t�=�P=#�
=#�
='�=0 �=0 �=49X=<j=L��=L��=P�`=aG�=ix�=ix�=u=�C�=�hs=�t�=�t�=�1=�
==�l�root|��������trrrrrr������� ����������������������������������������������UZfm������������zldU```ahnoz|������zna``�����
#<V^TQ[U</$
�������������������������)Bt���gR)�������������������������IEFR[t����������wp[I),6:@65) ����

�����������#"#&/<CHOUVUSMH</+%#������������������������
#<MRH/����������������������������������)-//-(��ghnt�����}trllhgggg����������������������������������������><BOS[gt��������t_G>��������������������������
�������")/7;HLJKHA;9/""""""XWW_dmz���������zmaX������������������������

������44345=BCJNPRTSNLB?54�������� �����)BN[\\ROEB5)5?GNgtqg^WVQB)"(6BO[diie`OB6)�����

������"/;HKKKH@;7/#"����
##
�����������������������������������������������

���������������� ������������


#+.,##
�����������������������������������������
#((#
����ZZahnzz���}znaZZZZZZ%&&(),))5BBBDDEB>5)%�������45.!�����135BO[h�������thOB61�������
#055'
���;HTahmz�����zmaTHHG;IFNQ[gilgd[NIIIIIIII��������������������)2BO[`dsvh[OB6)�����.24/)�����������������������������)+-120-)&������������ ������������������������������������������������gknz������znngggggg�a�n�zÂÇÉÇ�~�z�n�j�a�a�`�a�a�a�a�a�a�O�[�h�t�u�u�t�h�[�O�M�K�O�O�O�O�O�O�O�O�ûʻлٻܻ��ֻл������x�|�������������5�B�N�W�[�_�[�S�T�N�B�5�)�)�)�*�.�/�5�5�N�g�����������������������Z�(�����N�����������������������������������������H�a�z���������������z�H�"�����*�/�H�������������s�f�Z�M�I�C�A�M�Z�s���������7�����a�@�2�/�"�	��������������������O�\�h�j�u�ƇƉƁ�u�h�\�O�B�6�,�6�C�G�O�[�h�tčĖĘčā�h�7�)�����������G�[ÇÉÒÓÖÓÇ�z�z�u�zÂÇÇÇÇÇÇÇÇ��������������������������������������������
������������������������������������������������ŹŵŴŸŹ���������ҿ`�m�����������m�G�)�)�@�?�D�K�M�V�]�W�`�M�Z�f�s�t�s�o�b�Z�M�A�2�(����'�4�A�M��!�-�:�S�l�x�v�l�`�S�F�:�!���������_�c�_�\�S�P�F�:�-�(�-�8�:�F�S�Z�_�_�_�_E�E�E�E�FFE�E�E�E�E�E�E�E�E�E�E�E�E�EٹϹܹ����������ܹٹϹù����ùɹϹϹϹϾ�����������ʾƾȾ��������s�T�J�J�Z�s����������!�)�&�!��� ����������������"�.�;�G�O�N�G�;�.�"����������a�m�r�z�����z�m�a�W�T�S�T�X�a�a�a�a�a�aŔŠŭŹ��������žŵŭŔŇ�~�u�w�v�|ŇŔ���������������������������y�y�������������������
�����������ŻŹŶż���������6�C�O�h�q�u�}�u�h�b�\�O�C�6�+�*�(�*�2�6�����������ʾ׾����׾ʾ�������������������(�/�-�&��������ݿԿѿԿݿ�`�m�����¿����������y�j�`�T�G�;�1�=�T�`��������"�1�8�9�6�/�"�	���������������׾M�Z�f�h�s���q�f�Z�A�4������4�A�M�#�0�8�;�8�3�0�#���
��
����#�#�#�#�<�H�a�n�z�~�z�n�a�H�<�/�#������/�<���������������y������������������������ìù����������������ùìàÓÌËÏÓåì�L�Y�]�e�r�s�r�e�Y�S�L�A�L�L�L�L�L�L�L�L���ʼͼҼҼμ˼ʼļ����������������������ּݼ������������ּּӼҼּּּ�������������������������������������������(�5�A�J�U�[�Y�N�5��� ���������g�s���������������������x�s�k�f�e�f�d�g�zÇÏÓÖÓÓËÇÅ�z�w�s�x�z�z�z�z�z�z�������������
���������������������̻����*�2�2�)�%�$�����ܻŻ��������ܻ��'�@�������¼ü������r�Y�4�*���
���'DoD{D�D�D�D�D�D�EED�D�D�D�D�D�D|DsDlDo�h�j�f�q�t�y�|�t�p�h�[�O�A�>�6�9�B�O�[�h���ĽνнԽѽнĽ��������������������������!�4�:�O�S�_�c�_�S�:�-�!����� �������������ͺ��º����������������r�o�w���
��/�;�<�/�&�����������¸·���������
ìù����������������������ìàÊÓàâì���
��#�9�L�T�S�I�<�0��
���������������������������������|�y�l�j�d�^�`�l�s�y����(�5�A�L�V�_�b�Z�N�A�5�(�������EuE�E�E�E�E�E�E�E�E�EuEiE\EVEPEMEPE\EiEu���ɺ˺Ժֺ׺ֺɺ����������������������� )  A 8 � 8 S P N W E H D  F G V 6 u X \ 4 < ? F   u . \ D ' 4 5 M : H v 5 u = : d - D Z p X E T < A d S @ O ? U _ D -  �  �  N  /  �    D  �  �  �  �  R  ^  F  �  @  �  �  �  �  �  �  �  �  �  ]  �  �  U  �  �  9  �  9  $  �  i  �  9    �  �  �  W  �  :  E  �  �  �    �  K  I  �  5  �  �  �  ü��
�T��<ě���o<���;��
=C�<e`B=Ƨ�<�t�=�F=o<�C�=#�
=o=}�<��=��P<ě�<�`B<�/=��<�h<�`B<�`B=0 �<�/=L��<��=0 �=T��=��>�=L��=,1=��w='�=�t�=,1=T��=@�=<j=���=�\)=T��=aG�>@�>&�y>k�=��-=y�#=��=�{>J=�E�=�=�^5=��>hs=��#B
+Bt�B#'�BL�A�_[BB�%B| BecB�B
��B\�Bu�B��B+QBŎB��BBKBlHB->B��B��Bp�A�?4A���BngBO3B�WB�,B��B��BQB$ �A��B��B 
�B!P�B$JB"��B$�?BTB\�BHB�JB(-B�AB]�B��A��B��B*ÛB1�Bw�B��B�B,n~B(�B�"BU�B
:�BT|B#?�B�eA���B�B �B��B��BGBB�kB� B��BEB��BC�B>B��B@	BB��B�#Bx:A�_�A��wB�kB;WB�B¬B�{B?B �B#�xA�y	B@fB BYB!B$?B"��B%6�B��Bf'BR�B�-B@	BP�BK$B@A���B��B*�-B��B�WB|�B(B,@�B?B�	B��A��A��@��FA�|'A�yNA�:A���AC��A�A�B�sA���A�v�A�foA�\sA��_AiH8A<*@|@�3C�~>��AF��@`�Aa5�A�M A�n�Aru�A���B@AN��A�} AlluA��A:��A�Aä�A�8A��P?�^P@�'�A*A�ȳA��A��A�8�B-@���@��QC��A�k�A&��@tۓ@gfA��SA�"A�`A�-A�c/C���@-�EA�T�A��
@��RA�|�A���A�n�A�{�AC�A�|�B�3A׀�Aɀ�A�7�A�LA���Ak�A;{h@��@|�C�w�?4TAFg@c�QAa�}A���A�|�AppA�x�B ��AQ@A�hAjMlA��=A7�~A�XA�ADA�r{?��}@��dA�A�H|A�a�A�m�A�T�B�@��@�̠C���A�~yA&��@s�j@"�A���A�6A�}�A��A�r(C���@.�X   
   	   $                	   Y   
   m               /      :         	   /   	                           #   z         +      %               !      	   
   �   r   �               <      ,      )   &   
               /      =      U      /               )      !            -                              '      !      !                     !            1   /   )            #   %                                       !      E                                                                     '      !                           !            '   '   #                                 Nx//N��O	��N�K�O�q5N��zO�[�N�`oP���OBOg}oN.�N-�N�kO��O�2=O��O�0�Nk�N� �N�`1Oq��N���N��#N���O�]N��?O1O�N���OVP�O��xO�	�O ǿO��!N�=aO�T:N*i1OdM�8N���N��7NSaO��@O�w�N+�N�X
P�P?�Oͻ OT�kNbv�N㩞O|�fO�:�OD��O���N���O_8aO4اN��  �    M  �    �  �  �  �  �  
�  <  m  :  D  �  �  	@  :  U  �    Z  o  �  �  �  /  �  
  �  �       �  �  U  �  "  �  �  |  U  �  M    D    �  D  5  �  �  �  H  �  [  �  	l  ���h��1;o�ě�<t�;o<�C�<o<ě�<#�
=�+<D��<e`B<�1<�t�<��<�C�<�h<�C�<�t�<��
=�w<�1<�1<�9X<���<�j<���<���<�/=t�=o=�{=o=t�=,1=t�=t�=�P=,1=#�
='�=0 �=0 �=<j=D��=�{=��P=�v�=aG�=ix�=u=y�#=��T=�hs=�t�=���=�1=�
==�l�root|��������trrrrrr������� ����������������������������������������������]^aimoz��������zmda]```ahnoz|������zna``����
#<DE<5/#
�������������������������������)Bt���gK5������������������������srty������������}vts),6:@65) ����

�����������*)+/<HJQLHA<4/******�����������������������
#/<DJKKHF</#
���������������������������$*+*(%��ghnt�����}trllhgggg����������������������������������������[]cht����������thb\[��������������������������
�������")/7;HLJKHA;9/""""""][aelmz���������zme]������������������������

������44345=BCJNPRTSNLB?54�������� ���)5BDMSNC5)!5=ENgrpg]VUPB)%%')-6BOPVXVQOIB6*)%�����

������"/;HKKKH@;7/#"��������
 
����������������������������������������������

�����������������������������


#+.,##
�����������������������������������������
#((#
����]amnoz�{znaa]]]]]]]]&(()*15;BBCCB;5)&&&&�������������?<:<>BOh}���~th[OD?��������
#)+)!
���;HTahmz�����zmaTHHG;IFNQ[gilgd[NIIIIIIII��������������������)3BO[_bhqh[OB6)������(,-.)���������������������������)+-120-)&�������������������������������������������������������������gknz������znngggggg�a�n�zÂÇÉÇ�~�z�n�j�a�a�`�a�a�a�a�a�a�O�[�h�t�u�u�t�h�[�O�M�K�O�O�O�O�O�O�O�O�ûƻлѻһл˻û������������������������5�B�N�Q�[�[�[�N�I�B�5�.�1�2�5�5�5�5�5�5�N�Z�g�������������s�g�N�A�5�(���5�=�N�����������������������������������������T�a�l�|�~�v�y�m�a�M�H�;�3�:�9�A�D�D�J�T�����������������y�s�f�d�Z�R�R�Z�f�s��	�"�T�j�m�7�)�&��	�������������������	�O�\�c�h�u�}ƁƅƄƁ�u�h�\�O�E�C�6�6�L�O�6�B�O�[�a�a�[�P�B�6�)�������)�1�6ÇÉÒÓÖÓÇ�z�z�u�zÂÇÇÇÇÇÇÇÇ���������������������������������������������������������������������������������������������������ŹŷŶź�������ҿm�y���������z�m�`�T�G�>�;�5�9�D�L�R�`�m�M�Z�f�s�t�s�o�b�Z�M�A�2�(����'�4�A�M�!�-�:�S�X�p�o�g�_�S�F�:�-�!������!�_�c�_�\�S�P�F�:�-�(�-�8�:�F�S�Z�_�_�_�_E�E�E�E�FFE�E�E�E�E�E�E�E�E�E�E�E�E�EٹϹܹ����������ܹԹϹȹù��ù˹ϹϹϹϾ�����������������������s�f�]�^�j�s������������!�)�&�!��� ����������������"�.�;�G�O�N�G�;�.�"����������a�m�r�z�����z�m�a�W�T�S�T�X�a�a�a�a�a�aŔŠŭŹ��������ŹŮŭŔŇņ�z�{�{ŀŇŔ���������������������������y�y�������������������
�����������ŻŹŶż���������6�C�O�h�q�u�}�u�h�b�\�O�C�6�+�*�(�*�2�6�����������ʾ׾����׾ʾ�������������������(�*�)�"��������޿ݿ߿����`�m�����������������y�k�`�T�G�;�2�>�T�`�����	���!�%�#�"��	������������������M�Z�f�h�s���q�f�Z�A�4������4�A�M�#�0�8�;�8�3�0�#���
��
����#�#�#�#��#�<�H�U�a�n�p�t�n�a�H�/�#���������������������y������������������������ìù����������������ùìàÓÌËÏÓåì�L�Y�]�e�r�s�r�e�Y�S�L�A�L�L�L�L�L�L�L�L���Ǽʼмм˼ʼ��������������������������ּݼ������������ּּӼҼּּּ�������������������������������������������(�5�A�J�U�[�Y�N�5��� ���������g�s���������������������x�s�k�f�e�f�d�gÇÌÓÔÓËÇ�z�y�u�z�{ÇÇÇÇÇÇÇÇ�������������	�������������������������̼�� �$� �������ֻʻ»»Ļʻܻ���4�M�f���������������r�Y�@�4�%����'�4D{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D|DwD{�h�j�f�q�t�y�|�t�p�h�[�O�A�>�6�9�B�O�[�h���ĽνнԽѽнĽ�������������������������!�-�0�:�F�K�S�Z�S�F�:�-�!�����������������̺��������������������r�p�x�������
���$�$������������¿¾��������ìù����������������������ìàÊÓàâì���
��#�9�L�T�S�I�<�0��
���������������y���������������������y�l�l�g�a�l�u�y�y��(�5�A�L�V�_�b�Z�N�A�5�(�������EuE�E�E�E�E�E�E�E�E�EuEiE\EVEPEMEPE\EiEu���ɺ˺Ժֺ׺ֺɺ����������������������� )  ) D d 8 d A I R  H <  J L V 8 u X R  < ? F ! u . \ D   3 ) M : = v 5 u 5 : d - D J Y W A R < A [ M 3 O ? E _ D -  �  �  <  �  V    B  +  �  P  �  R  7  �  o  _  �    �  �  �  �  �  �  �    �  �  U  �    #  W  9  $  1  i  �  9  �  �  �  �  W  [  �    j    �       *  r  �  5    �  �  �  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  �  �  �  �    u  j  _  T  G  :  -         �  �  �  �  �    �  �  �  �  �  �  �  �  ~  g  P  9       �  �  �  _    �  �  �    )  ;  I  M  E  /    �  �  l    �  `    �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  e  J  0  �  �  �              �  �  �  �    K    �  K  �   �  �  �  �  �  �  �  �  �  z  n  a  T  G  9  '       �   �   �  3  B  M  Z  u  �  �  �  �  �  �  k  3  �  �  l  [  "  �  ;  �  �  �  �  �  �  �  �  �  �  �  �  �  x  m  ^  M  7     �  �  s  �  �  �  �  w  5  �  �  �  n  C    �    `  �      �  �  �  �  �  �  �  �  �  �  t  Z  @  %    �  �  �  ,  t  ?  �  �  	   	S  	P  	�  
  
z  
�  
�  
�  
>  	�  	w  �  U  4  M  �  <  �  �  �  �  �  �  �  �  [  -  �  �  �  ^  $  �  �  j  '  _  b  e  h  k  m  l  l  k  j  g  a  [  U  O  I  B  <  6  /  �    !  -  4  9  9  1  !    �  �  �  C  �  �  O  �  �  E  ;  @  C  D  B  =  4  "    �  �  �  �  y  _  D    �  �  K  �  �    J  h  z  �  ~  h  7  �  �    �  %  �    x  �  �  �  �  �  �  �  �  �  t  �  s  \  C  &    �  �  �  x  ^  S  �  	  	&  	9  	@  	8  	  �  �  g    �  >  �  L  �  .  �  T  �  :  N  b  n  g  `  V  K  @  3  %    @  a  p  g  ]  S  G  <  U  A  /  4  5  &      	  	      �  �  �  �  �  �    @  w  �  �  �  �  }  p  d  V  H  :  +      �  �  �  �  �  �  �  �     >  t  �  �  �     �  �  �  �  �  @  �  z  �    M  Z  L  >  2  (      
  �  �  �  �  �  �  �  n  Z  X  t  �  o  l  j  f  `  Z  S  L  F  <  2  '        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  X  @  &    �  �  �  �  ^  �  �  �  �  �  �  �  |  i  U  @  )    �  �  �  �  y  3  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  l  Z  H  6  /      �  �  �  �  k  >    �  �  s  :  �  �    f  �   �  �  �  v  l  a  U  I  <  .      �  �  �  �  �  �  �  l  X  
  
    �  �  �  �  �  �  �  �  ~  y  \  +  �  �  O   �   �  �  �  �  �  �  �  �  �  �  �  �  ^  4    �  s    �  
   �  �  �  �  �  {  ^  `  a  o  }  �  n  K  !  �  �  K  �  6  j  d  �  ;  �  M  �  �            �  f  �  �     �  
�  	R  �     �  �  �  �  �  �  |  i  R  7    �  �  h    �  U  +   �  �  �  �  �  �  �  �  �  z  f  N  2    �  �  �  �  �  �  |  �  �  �  �  �  �  �  m  /  �  �  H    �  ]  �  {  �  u  �  U  J  @  5  *      �  �  �  �  �  �  �  s  ]  E  .     �  �  �  �  �  �  �  �  ~  �  �  |  c  @    �  ~  8  �  Y  p  "               (  (        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  g  G  "  �  �  �  ]  >  .  �    y  q  f  [  K  9  &    �  �  �  �  �  �  �  �  �  �  |  y  v  r  o  k  f  b  ]  W  Q  K  E  >  8  2  }  �  ;  �  U  S  N  J  N  L  E  :  -      �  �  �  D  �  �    �  �  �  �  �  �  �  �  w  W  8    �  �  �  k  ,  �  �  ^  �  �  -  1  6  =  F  M  N  O  C  7  "    �  �  �  ~  T     �  �                �  �  �  �  e  /  �  �  �  {  m  P  3  �  p  �    ;  C  4    �  �  I  �  N  �  
�  
  �  �  X  �  
T  
�  
�  
      
�  
�  
�  
�  
b  
-  	�  	�  	  �  �  �  �  �  [  �  B  ~  �  �  �  5  �  -  \  `  <  �  y  s    	�  �    D    $      �  �  �  v  D    �  �  �  m  9  �  {     �  5  *      	  �  �  �  �  �  �  �  �  �  �  {  j  Z  I  9  �  �  �  �  �  �  �  �  �  }  `  >    �  �  �  X  *    �  �  �  �  U  3    	  �  �  �  l  1  �  �  w  D    �  c  �  s  �  �  �  �  �  �  q  B    �    /  �  ^  �  �  �  �  =  H  7  "    �  �  �  e  3  �  �  �  O    �  �  9  �  �  0  �  �  �  �  �  �  t  I    �  �  2  �  h  �  �  #  �  E  u  !  H  U  [  W  S  K  C  7  $    �  �  x  ?    �  t  7  �  �  �  �  �  �  o  ;    �  �  _  &  �  �  Z    �  c    �  	l  	D  	  �  �  �  �  v  a  B    �  �    �    S  Q  G  �  �  t  \  B  (    �  �  �  �  x  X  8    �  �  h    �  �