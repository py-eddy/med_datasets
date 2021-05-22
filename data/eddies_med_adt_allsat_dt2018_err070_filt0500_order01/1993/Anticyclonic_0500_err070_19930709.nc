CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�r� ě�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�sd   max       P�*�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �e`B   max       =�F      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @E��G�{     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @vt          �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @L�           p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @���          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ě�   max       >�V      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��m   max       B+��      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��K   max       B+�2      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C�i�      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�   max       C�g�      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�sd   max       P�|B      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���u��"   max       ?��
=p�      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �49X   max       =��m      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @E��G�{     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vt          �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @L�           p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @�I�          �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >   max         >      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��t�j~�   max       ?��
=p�     0  O�                                 #   	               2   
   g            /                  N               J            3            	            E                  $   $   �   "O�2�N��N>^N:��O��4N��zO��{O.Or;zN>t�OJh�N��OxN}WxNݐ�N��;P�7N��P!lVNW�N�2�O�PC�OC�N���O���O�eN&��P0x�O�ޔO�"N�&�M�sdPz�N(�cN�{ N��P8[OJ	O�>NWdaN}a�OD��O7N!��P�*�OS�O�n
N��JN�IO9��Oa)�O��5O;�O5�=�e`B�D���49X�49X�o��`B�ě����
��o��o%   :�o;o;��
;ě�<t�<#�
<#�
<49X<T��<�o<�t�<�t�<�t�<�t�<�t�<��
<��
<�j<�j<�j<���=C�=C�=C�=\)=\)=�P=��=��=��=�w=�w=#�
='�='�=ix�=}�=}�=�%=�+=��=���=�9X=�F=;876BN[glrvvsng[NB=������������������������������LOQ[[[hnlh[OLLLLLLLL#0IUbfhfbUI<7.,&������ 

�����RSVX[gt������ytmge\R��������������������de]]dt�����������thd��������������������  #/<HLU][UUH</(#��������������������fhknvz������������tf����������������������������������������~���������������~~~~������	+58$����*&'/<HIUXUSMHD><94/*��������������������),6766)��������������������?BCLNU[bgijjg^[SNJB?���#/3BUgaZR</
����������
#-+)&#
��	
#'/15/*#
				ipoqw~�����������zni������
#+1563/#
�������������������������������������������)5BIKB5)��������

�������������������������\Z_abmonma\\\\\\\\\\�BNg�������t[
� #08<10#          �������
�������
	)-*)!
)7BEX^~tynO6)(0.*)15>A;5)����������������������������������������# #$/2:<==<<;/(#####|������������������|:8<?DIU`bcgjgdbZUI<:�����������������������)6FSP6���������������������������)6BO[aa[O;7)������������������������������������������������������������z�����������������zz������)141)������������

�����������������������B�O�[�h�tĀăā�y�t�h�[�O�B�6�0�1�0�7�B��)�6�B�J�L�B�6�2�)������������������������ýùìçìùÿ���������������#�/�/�/�+�#�������������������������������x�_�S�F�F�C�F�S�_�l�����ûлܻ���������ܻлû�����������������������������������������������������Ŀ������ĻĳİĦĚčĂ�~āćčĚĥĦĳĿ�/�;�H�T�a�g�`�_�T�O�H�;�/�*�"����"�/�������������������������������������������������������������������T�V�a�i�h�a�T�H�G�A�H�P�T�T�T�T�T�T�T�TE�E�E�E�E�FFFE�E�E�E�E�E�E�E�E�E�E�E��#�/�5�<�H�O�H�A�<�/�#����#�#�#�#�#�#����������������������������������ú���ſ����������������������������������������/�;�H�a���������������\�T�/����	��/�����	������	���������������������ùܹ�����'�,�-�'�����ܹ����������ÿy���������������y�y�n�x�y�y�y�y�y�y�y�y�.�;�G�J�M�G�;�7�0�.�"����	��	��#�.�����������������������ƹ�����������(�5�V�a�Z�A�5��
�����ݿǿ��Ͽ޿���(�G�T�`�m�y�������������y�m�`�T�H�K�G�D�G��"�+�-�.�8�.�$�"���	��	�
��������(�5�N�Z�s�y�~�g�Z�M�A�(�������������5�=�:�5�%�������������������ÇÓàèáàÓÓÒÇÂÆÇÇÇÇÇÇÇÇčĚĦ�������������ĿĚā�w�s�s�x�yāč�
��#�0�<�I�O�R�N�G�9�#�
����������
�4�A�Z�f�s�w����x�s�f�Z�M�A�7�4�5�4�1�4��������������������޽����������������������������������������������������������������s�g�[�[�U�L�P�Z�{���ּ������ּּӼּּּּּּּּּֿ"�.�6�;�G�H�G�D�;�.�"������"�"�"�"���ʾ׾����	��	�������׾־ʾǾ������Y���������f�Y�4����һ����ûܼ�4�Y�y�����������y�v�m�`�T�M�H�P�T�`�m�m�y�y�4�A�H�M�Y�T�M�A�2�(���������(�4�/�<�D�>�<�/�/�#� ��#�&�/�/�/�/�/�/�/�/�s�����������z�s�f�Z�W�Z�a�f�q�s�s�s�s���ûƻлֻݻ߻ܻٻлû���������������������������������ɼ�������������{�{�|��a�m�z�}�z�q�m�a�Z�]�a�a�a�a�a�a�a�a�a�a�/�T�X�W�P�H�4�"�	�������������������	�/������(�4�A�M�S�V�V�M�4�(��������~�����������ͺǺȺ����������~�n�g�c�r�~Óàìó÷ùúùìàÓÑÉÈÓÓÓÓÓÓ�	�	���������
������	�	�	�	�	�	�	�	�l�y���������������������y�p�`�Z�[�d�`�l�����ʼּ��������ּʼ����������������!�-�:�F�H�V�`�d�c�V�F�:�-�*�"�����!D�D�D�D�D�D�D�D�D�D�D�D�D�D�D~DrDsDD�D�E�E�E�E�E�E�E�E�E�E�E�E�E�EzEsEqErEuE�E�   D f ; 1 s ) Y % 2  4   6 U \ H 9  C C B G $ : Z @ A * > : b M M > + D h : Y a ] > ; T ? ` s ? C  C ?  V    �  �  w  _  `  �  �  a  �  W  �  �  �  �    �  $    �  W    m  ]  �  �  l  L  E  
  �  k      y  C  �  (    &  �  R  �  �  B  )  �  �  �  �  *  �  �  �  �  �<u���
�D���ě�;ě��D��<�1;ě�<e`B;o=\)<t�<�<�t�<�9X<T��=u<���=�x�<���<�1<ě�=��=�w<�9X=#�
=,1<���=���=,1=�w=o=�P=�/=�w=0 �=0 �=�9X=H�9=P�`=8Q�=@�=u=e`B=<j=�G�=���=�{=���=�C�=��=�/=�S�>�V>��B��B&�B͝BHzB&i.B#�^B	��B��B
2�B�RB�(Br"B�B":�B�BB%BO}B̆Bf�B*B�tB�wB�B�B��B�B-0B��BUBVEB#�yB ��A��mBB%��Bi;BcB��B�B�B�BDQB��B'�BfB`�Bd�B��B!.|B5B+��B<B��B�\BNB��B<NB?"BBwB&=/B#ŤB	lB�eB
�B��B��BI�B?�B"��B�bBG<B@XB�B@B:�B�9B�B��B0DBq B?�B�B��B�QB;B#��B �0A��KB	�%B%r&BA)B�
B��B�@B�%B��B@SB �B':�BEB�B�B��B!?�B;�B+�2B8+B?�B�WB�~A�X�A֎�A�N:A���@�X"@��GA�=|A���A���A�FsAң*A���C�i�A�y�A�u�As�5A��QA���?��An� A`4<B;�A���Ak-�A^!A�A��@Aʃ�A��.A�5�A>��A0mA�j�A���A#�A`zAT�@���AjV�A7x�A�x�AC��@�4@�n@A�r�A�xYA6g(@��A��A�Z�A$@���@x*�C��!C��A�e�A�v�AΆ(A���@�#w@�i�A���A��dA�$�A��]A��A���C�g�A��AІ�At	�A��A��>�An�A^9B5�A�q}Ak ~A]<)A�}�A�v�AɁ�A��A�0�A?A/��A�I.A�e�A�A`�IASU�@�xhAiK�A7'�A�K@AC�7@��@��A�{�A��qA7P_@�AːXA�|�A��A ��@lDC��C�	         	                        $   	               3   
   h            0                  N               K      	   	   4            	            F                  $   %   �   "                                                   -      )            3                  -   !            3            ;                        =      #                                                                                          !                                 )            7                        =      #                     OQ�dN�HN>^N:��O��4N��zN��(N�&�O��N>t�N�7ON��OxN	½N��fN��;O���N��2O~[NW�N�2�O�O�[�OhN���O�1	O�eN&��OZ�O�R�N�ݶN�&�M�sdP*�FN(�cN�{ N��P*�OJ	O�>NWdaNG��OD��N�u�N!��P�|BOS�O�n
N��JN�IO9��O2;^O�s*O�C�OsM  o  z    %  �  A     �  �  X  L  �    �  �  �  �  �  
�  �  �  m  ?  �  �  �  �  �  �  F  l  .  �  
  a  �  �  �  �  �  �  �    O  �  O  '  j  �  �  c    �  2  �#�
�49X�49X�49X�o��`B;ě��D��;D����o<t�:�o;o;�`B<t�<t�<�<D��=q��<T��<�o<�t�=�P<�j<�t�<���<��
<��
=y�#<ě�<���<���=C�=H�9=C�=\)=\)=�w=��=��=��=#�
=�w='�='�=49X=ix�=}�=}�=�%=�+=��-=��-=�/=��m@>;;;BN[gipttphg[NB@������������������������������LOQ[[[hnlh[OLLLLLLLL#0IUbfhfbUI<7.,&������ 

�����Y[`gt������tmg^[YYYY��������������������mijhit�����������tmm��������������������!#*/<CHNUQHA<4/&&#!!��������������������fhknvz������������tf����������������������������������������~���������������~~~~������%&$ ���,(+/<HTNJH<<;/,,,,,,��������������������),6766)��������������������?BCLNU[bgijjg^[SNJB?�����
#=BNKA</!����
#('&##
����	
#'/15/*#
				jrprtx����������znj������
#+1563/#
��������������������������������� ����������)5BHJEB5)���������	


��������������������������\Z_abmonma\\\\\\\\\\BNgt�������[B5) #08<10#          �������
�������
	)-*)!
)6@CV\|rtk]B3.1/+)15>A;5)����������������������������������������!#(/18<<<</,$#!!!!!!|������������������|;:<>@EIUbfifcbYUI<;;������������������������)DQO6���������������������������)6BO[aa[O;7)�������������������������������������������������������������������������������������*-/)�����������

�����������������������B�O�[�h�t�{��}�v�t�h�[�O�B�6�5�6�6�>�B��)�6�B�A�6�/�)��������������������������ýùìçìùÿ���������������#�/�/�/�+�#�������������������������������x�_�S�F�F�C�F�S�_�l�����ûлܻ���������ܻлû�����������������������������������������������������čĚĦĳľĸĳĩĦĚčĆāĉčččččč�"�/�;�H�T�`�V�V�T�H�E�;�6�/�"�!���"�"������������������������������������������������������������� ���T�V�a�i�h�a�T�H�G�A�H�P�T�T�T�T�T�T�T�TE�E�E�E�E�FFFE�E�E�E�E�E�E�E�E�E�E�E��#�/�1�:�<�@�<�/�+�#�"�!�#�#�#�#�#�#�#�#���������������������������������������ҿ����������������������������������������/�;�H�T�a�o�v�v�l�a�T�H�;�6�(�"���"�/�����	����	��������������������������Ϲܹ������������ܹƹù������ùϿy���������������y�y�n�x�y�y�y�y�y�y�y�y�.�;�G�J�M�G�;�7�0�.�"����	��	��#�.�����������������������ƹ�������������5�A�D�C�@�6������ݿؿֿݿ�����m�y������������y�m�`�T�S�S�R�T�`�b�m�m��"�+�-�.�8�.�$�"���	��	�
��������(�5�A�N�Z�g�v�u�Z�N�A�(�����	��������5�=�:�5�%�������������������ÇÓàèáàÓÓÒÇÂÆÇÇÇÇÇÇÇÇčĚĦĳĿ����������ĿĳĦĚďčĈĆčč��#�0�<�I�N�Q�L�E�7�#�
�������������A�M�Z�f�q�s�{�s�q�f�Z�M�A�A�9�;�A�A�A�A��������������������޽��������������������������������������������������������������s�i�d�c�f�a�`�j�s�~���ּ������ּּӼּּּּּּּּּֿ"�.�6�;�G�H�G�D�;�.�"������"�"�"�"���ʾ׾����	��	�������׾־ʾǾ��������������f�Y�4����ܻǻûлܼ�4�Y��y�����������y�v�m�`�T�M�H�P�T�`�m�m�y�y�4�A�H�M�Y�T�M�A�2�(���������(�4�/�<�D�>�<�/�/�#� ��#�&�/�/�/�/�/�/�/�/�����������v�s�f�c�d�f�s�u���������ûƻлֻݻ߻ܻٻлû����������������������������������������������|�|�~���a�m�z�}�z�q�m�a�Z�]�a�a�a�a�a�a�a�a�a�a�;�H�T�W�V�N�F�"�	�����������������	�"�;������(�4�A�M�S�V�V�M�4�(��������~�����������ͺǺȺ����������~�n�g�c�r�~Óàìó÷ùúùìàÓÑÉÈÓÓÓÓÓÓ�	�	���������
������	�	�	�	�	�	�	�	�l�y���������������������y�p�`�Z�[�d�`�l���ʼּ߼��������ּʼƼ������������!�-�:�G�T�^�`�`�S�F�:�-�%�!������!D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DwDxD�D�D�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E|EuEuEuE{E�  ; f ; 1 s % A ' 2  4   f 1 \ + )  C C B V   : V @ A  ? - b M T > + D g : Y a F > 7 T 8 ` s ? C  ! A  T    �  �  w  _  `  �  �  �  R  W    �  �  J  �  �  �  �  �  W    m  �    �  '  L  E  �  �  �      J  C  �  (  �  &  �  R  c  �    )  ~  �  �  �  *  �  u  K  j  d  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  8  T  i  l  `  N  :  %    �  �  �  ~  F  
  �  d  �  N    j  p  u  x  r  m  a  R  B  )    �  �  �  �  �  m  M  *      �  �  �  �  �  �  d  L  5  G  �  �  �  }  `  B  &    �  %          �  �  �  �  �  �  �  �  �  �  �  {  e  O  9  �  �  �  �  }  n  ^  N  =  &    �  �  �  �  c  d  m  p  e  A  9  1  )  !          �  �  �  �  �  �  �  �  �  �  �  )  [  �  �  �  �               �  �  i    �  <  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  X  5  �  �  ^  w  �  �  �  �  �  �  �  �    w  m  a  T  F  1  
  �  X  X  O  F  <  3  *  !                $  0  ;  G  S  _  �  �    5  E  L  H  7      �  �  �  p    �  ;  �  �  @  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Z  2    �  �    �  �  �  �  �  �  �  �  �  �    �  �  k    w  �  +   x  l  r  {  �  z  l  ]  T  �  �  �  ~  w  o  e  [  Q  E  9  ,  /  �  �  �  �  �  �  �  �  p  \  D  )    �  �  �  �  �  h  �  �  �  �  �  �  �  �  �  �  |  l  ]  N  ?  (     �   �   �  +  �  �  �  �  �  �  �  �  �  �  �  �  R    �  T  �  �  Y  �  �  �  �  �  �  �  �  �  �  �  �  |  o  `  P  B  3  &    Q  �  	u  	�  
  
C  
o  
�  
�  
�  
�  
{  
.  	�  	A  �  r  #  �  $  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  u  o  o  o  o  j  `  U  J  <  *       �   �   �   �  m  e  ]  U  I  =  0      �  �  �  �  �  �  u  a  M  8  $  �  �  �      +  8  >  >  0    �  �  a    �  S  �  �  �      o  }  �  �    o  \  I  /    �  �  i     �  ~  [  ]  �  �  �  x  o  f  Z  M  A  4  &      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  V  )  �  �  �  =  �  }  �  {   �  �  �  �    j  S  :      �  �  �  x  Z  ,  �  �  q  �  �  �  �  �  �  u  o  i  c  ^  [  X  U  R  O  L  I  F  C  @  >  p  �    o  �  �  3  _  {  �  �  t  R    �  )  �  �    �  B  F  F  C  :  .       �  �  �  �  �  \  )  �  �  z  �  z  S  ]  f  k  k  g  b  V  E  0    �  �  �  �  S    �  q    .  *  %  !            �  �  �  �  �  �  �  �  y  l  _  �  �  �  �  �  �  �    g  O  6      �  �  �  �  c  B     ]  �  �  �      �  �  �  q  J  �  h  0  �  P  �  �  �  �  a  c  e  g  g  a  [  U  K  <  -      �  �  �  �  �  {  ^  �  �    w  n  d  W  J  9  '    �  �  �  �  �  K     �   �  �  �  �  z  t  l  ]  M  ;  (    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  _  :    j  X  -    �  x  �    �  �  �  �  �  �  �  �  ~  c  G  )    �  �  �  E    �  e    �  �  �  �  �  �  �  �  p  S  4    �  �  �  �  Z  &  �  �  �  �  �  �  �  �  �  �  �  �  �  v  i  ]  P  B  3  $      �  �  �  �  �  �  �  c  G  +    �  �  �  �  f  <  	  �  �    �  �  �  �  �  �  �  v  W  7    �  �  {  B    �  6  �  +  J  I  F  B  9  )    �  �  �  �  �  �  �  {  e  L  1    �  �  �  �  �  �  �  o  b  [  U  O  @  *       �  �  �  �  3  N  C  1  .  &    �  �  �  d    �  V  �  �  �    �   �  '  "        �  �  �  �  t  J    �  �  �  O    �  �  X  j  O  *    �  �  �  �  �  �  �  t  8  �  �  \  -    �    �  �  �  �  h  9  	  �  �  �  ^  >    �  �  �  �  o  6  �  �  �  �  y  f  T  B  0      �  �  �  �  t  V  7     �   �  c  O  C  <  4  ,    �  �  �  �  �  d  <  
  �  �  *  �  �  �          �  �  �  �  �  g  .  �  �  H  �  �  3  �  y  �  �  �  z  v  �  }  a  <    �  �  ^    �  `  �  ,  q  '    )  2  '    �  �  �  I  �  �  �  S  a    �  �    �  �  �  �  �  �  �  �  z  C    �  �  =  �  }    �  �  ?  l  �