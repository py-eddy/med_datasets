CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�S����      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��U   max       Q`�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       >+      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���
=q   max       @F"�\(��     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��fffff    max       @vhQ��     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P            t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ͱ        max       @�<`          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��/   max       >�-      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��	   max       B0�      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��Z   max       B0/�      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =��   max       C��      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =�2   max       C�-      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         .      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          O      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��U   max       P�q�      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��䎊r   max       ?��&��J      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       >.{      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @F"�\(��     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����R    max       @vg
=p��     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P            t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ͱ        max       @�$@          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         G   max         G      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?r�s�PH   max       ?��&��J     �  QX   
         
                        i         R      
         6                  @             7   
   '   /          0   ?     .   [         
            \   !            M   	   {         N�0�N7�M��UN-�gO��N���N��N��N�8�O�<�O�:Q`�O
׽N���Pe�$N���N�P@N%4OV�P��N�
Oɹ~N��N;:Nn�O��O��xN [�N�O�tO	�\O8��O��OЛ�O�r�O~7�P#^�O+��P\F>O��]N�6N@��N�-pO%��O%�TN���P\�fO��O��RN�&N�)�O�h�NY+O��O���OE8�N\����������1���
���
��o%@  %@  ;��
;�`B<#�
<#�
<D��<e`B<e`B<�o<�t�<��
<�j<�j<�/<�h<�=o=C�=�P=�P=��=#�
=#�
=#�
='�='�=0 �=0 �=49X=<j=@�=@�=@�=H�9=L��=T��=q��=y�#=��=�O�=�hs=�hs=���=�E�=�Q�=��=���=���>+31368BHOOOOGB6333333otz|��������tooooooot{��������toooooooo���������������������	"/;FOPWTH;/)"	��������������������rryz��������~zrrrrrrdab]\ghtt���~ztgdddd').6BOVZOJDB61))''''����
&**&#���������������������������[hlktztg[B������������������������fceht�����thffffffff�����
./4+ ����������� ����DHUacnpppnjaUROONHDD>>?BOSOOIB>>>>>>>>>>*26=CGKKFC6*%JDDJT[ht��������h[OJz�������������zzzzzz����)65;51,$���qlity�������������tqNEOT[hlhf_[ONNNNNNNN?>BO[^[VOB??????????�����
/<CMNG<#
���������5BR[[NB5)�./<HKIH<7/..........����������������ebdlt�������������te�����
"!
��������������������������hgmz�������������zmh���������������������������� $#
���)<@BFLB:.)������)9DF5)������������������������( *5Ng���������gN5(MIUenz���������zkZUMttx�����������vttttt����������������������������������������]ZZ_afmz}������zmha]��������������������""/17;7/"�������)052"�������������

�������������������������������������������� 	
##(+/#
�����

��������*.51*��������� 
���������������������������������	�����xvz�������zxxxxxxxx�l�y���������������y�x�l�a�b�l�l�l�l�l�l�zÁÇÓßÓÎÇ�z�u�z�{�z�p�z�z�z�z�z�z�����ûɻлһлûû��������������������������������������������������������������T�a�i�m�r�s�s�m�a�T�;�/�!��#�/�;�C�N�T�����������������������Ŀѿݿݿ�ݿѿѿĿ����������ĿĿĿĿĿ���#�0�<�I�I�I�<�<�1�0�+�#������������������������������������������������N�Z���������������s�g�Z�N�H�A�5�2�5�A�N��"�/�;�T�\�a�g�`�T�H�;�/�"�	����������-�+�������Ƨƍ�h�6���!�C�C�OƁ��������*�6�?�>�6�*�$���������������F�S�_�d�h�e�_�S�F�@�=�B�F�F�F�F�F�F�F�F�5�B�h�z�t�n�[�B�6�)�����������������5���������������������|�x�l�_�[�_�n�x�����	����	� ������������������������	�	�ܻ���������ܻջܻܻܻܻܻܻܻܻܻܿ`�m�y���������������y�m�`�T�Q�N�Q�T�\�`��A�Z�f�s�x�}�}�w�f�A�(������������H�N�R�M�H�<�7�/�+�$�'�/�<�>�H�H�H�H�H�H�.�;�G�T�b�c�`�b�T�G�;�"��	��������.�T�`�m�n�m�k�k�m�y�|�y�q�m�`�T�P�L�J�Q�T�<�H�N�U�[�\�U�K�H�<�7�;�<�<�<�<�<�<�<�<�ѿݿ���ݿؿѿſ̿ѿѿѿѿѿѿѿѿѿѿ`�m�y���������������m�`�G�@�5�7�;�E�T�`�y�������������������|�x�_�S�M�L�P�V�`�yEiEpEsEkEiE\E[EWE\EgEiEiEiEiEiEiEiEiEiEi�A�M�Z�^�f�s�}��s�f�Z�M�A�?�4�0�4�6�A�A�����	���'�-�/�-�1�/�"����������������;�H�T�X�^�a�b�c�a�T�H�;�6�4�3�1�;�;�;�;ÇÓàì÷÷æàÓÑÇ�z�n�h�k�n�w�zÄÇ������������������ùìàÛÕÓØâ÷���Ž����Ľн���
�����н��������������������������þʾҾɾ�������f�e�j�f�b�f�r���r�������������v�r�f�b�Y�M�J�K�R�Y�f�r��������������������������������������������'�.�1�)�'��������߻޻���������6�>�F�F�=�"�� ��������������빶�ùϹܹ���ݹҹù��������������������@�B�M�R�U�Y�T�M�@�4�*�*�,�.�4�=�@�@�@�@�)�/�6�<�6�-�0�)�(�����"�)�)�)�)�)�)�������
������
��������������������ŠŭŹ����������������ŹŭŠŝŔŒŔřŠ�лܻ������������ܻлɻû����ûͻ����#�0�<�=�?�<�0�#�����������������ɺʺ����������~�r�Y�1�-�5�?�Y�~���/�<�H�a�n�zÁÀ�z�x�n�a�H�<�/�#���#�/���������������������������������u�������N�Z�_�g�r�o�g�Z�O�N�A�;�A�D�N�N�N�N�N�N�*�6�C�O�P�T�O�C�B�6�*�����!�*�*�*�*E�E�E�E�E�E�E�E�E�E�EuErEnEmEmEuE�E�E�E����������������y�w�v�y������������������DoD{D�D�D�D�D�D�D�D�D�D�D�D{DoDbDaDbDhDoĦĳĿ��������������ĿĳĦěĚĖĔĖĚĦ������
���#�*�/�#��
����������������ݽ�����ݽнĽ½Ľн׽ݽݽݽݽݽݽݽ� ; n � W D X + O @ L - ; C /  C I d ' " I ' j S > - C P U J G K 8 6 R  $ L G ; M X =  ' / L E G @ < A D  ) K o    �  d  h  ^  T  �  �  �  �  p  6  	[  A  �  �  ,    ?  1  �  �  �  )  ]  B  �  �  7    V  R  �  �  �  �  �  �  t    �  D  �  "  _  l  �  �  !    �  �  N  t  �    �  0��9X��/��9X�D�����
;o;ě�;ě�;��
<�/<�9X=�x�<�9X<���=\<�`B<���<���=C�=���=t�=e`B=C�=�P=C�=Ƨ�=�C�=D��=P�`=\=L��=���=�-=�t�=u=�^5=��=��>�->I�=y�#=�o=q��=�+=��P=�C�> Ĝ=��=��-=���=�"�>)��=ȴ9>dZ>1'>   >O�B"	B
*B�B"T�A�Y]B '�B &`B	pyB�B�xB�sB�wB�BadB��B��B��B^[B0�BT�BiB+�B#B.'B��B��B\�B�tB#\�BR�Bz[B"4�B ��B!�B�B�BxGB!�lB˳B�	Bq�B�7BbA���BJ�A��	B�B�B�B�9B>�B5B/d�B��B��B��B��B;�B	íB�)B"AtA�l)B O\B 9B	�wB�wB2B�B5B@�BDVB�B�YB@B>�B0/�BK�B�ZB�NB�'BEB� B��B@�B�B#;^B
�6B��B"?�B?�B">B�B��B¡B!��B�B�7B�2B��B:)A���Bu.A��ZB>�B;2B��B��BA(B@$B/Q-B�CBi�B�`B�A��A�WH@�m�@�A��*@T��Ay�oA�ZPA�f�A���A��B��A��@�ȈA�â@�E�A���@�G�AklA9��A�w Aa�&Aj9_AĴ�A|�Ajp�An�]C��&A>~�A�s&A�o�A�l3A�A''AH�k@�x)A�a9@��A��=��@�#�A֞ A��dA��@��A��@��A�&A���A�BB 4�C��AnQC���A��cA�oA*0oA�.A�{�@�l@�0A�\�@V^�Az��A�A�}A�DrA��pB�A�B@�NA�^z@�- A�@�|Aj�A8��A�b(Aa*�Ai)A�oA|'�Aj,Am^�C��|A=��A��CA���A�{bA���A&�0AJoq@���A�~@�D�Aӛ�=�2@���A։@A�vsA��@�@A�^>@��A�{�A��A�o(B ��C�-AܷC��1A��A��A)��   
         
                        i         R               6                  @             8      (   /          1   ?     .   [         
            ]   !            N   	   {                                       !      O         1               %      %            !   %         #            #   !      '      1                        3                                                                  5         '                                    #         !            #         '                              '                              N�0�N7�M��UN-�gO�)eN���N��N��N�8�OX��OkS�P�q�O
׽N���P+L%N���N�P@N%4OV�O���Nu�O���N��N;:Nn�O�&O�N [�N�O���O	�\N�""O�xOЛ�O�UtO7P�KO+��O�C�Oh�UN�}N@��N�-pO%��O��N���O�l�O��O��RN�&N�]�O*}�NY+OO5�O[3OE8�N\  �  �  b  �  E  f  �  �  �  ?  �  "  �  �  �  o  �  �  �  �  C  /    �  b  	m  �  .  �  {  �  �  �    �  �  �  0  �    �  #      V  �  
d  �  v  )  �  /  �  �  0  �  ڼ���������1��t����
��o%@  %@  <o<t�=T��<#�
<D��<�h<e`B<�o<�t�<��
=49X<���<�<�h<�=o=e`B=��=�P=��=49X=#�
=D��=,1='�=49X=m�h=H�9=<j>.{=��=T��=H�9=L��=T��=u=y�#=�v�=�O�=�hs=�hs=�-=�"�=�Q�=��=�"�=���>+31368BHOOOOGB6333333otz|��������tooooooot{��������toooooooo�������������������� 	"/;@HKMTPH;/%"��������������������rryz��������~zrrrrrrdab]\ghtt���~ztgdddd').6BOVZOJDB61))''''�����	#((%# �����������������������������=KMKSNB5�����������������������fceht�����thffffffff�������
$(& 
��������� ����DHUacnpppnjaUROONHDD>>?BOSOOIB>>>>>>>>>>*26=CGKKFC6*%VUTVY[ht��������th[V~�������������~~~~~~�����)251.)����qlity�������������tqNEOT[hlhf_[ONNNNNNNN?>BO[^[VOB??????????�� 
#/4<@CFB</#
�������15@OZNB5)�./<HKIH<7/..........����������������dent��������������gd�����
"!
��������������������������kimz�������������zmk����������������������������

���
	")26865-)�����)7BD3)������������������������4114BN[gt����{tg[NB4[[cnz����������znda[|z������������||||||����������������������������������������]ZZ_afmz}������zmha]��������������������""/17;7/"������������������������

��������������������������������������������
!#$'&#
�������

������*.51*����������	���������������������������������	�����xvz�������zxxxxxxxx�l�y���������������y�x�l�a�b�l�l�l�l�l�l�zÁÇÓßÓÎÇ�z�u�z�{�z�p�z�z�z�z�z�z�����ûɻлһлûû��������������������������������������������������������������T�a�g�m�q�r�p�m�f�a�T�;�/�%� �'�;�>�E�T�����������������������Ŀѿݿݿ�ݿѿѿĿ����������ĿĿĿĿĿ���#�0�<�I�I�I�<�<�1�0�+�#������������������������������������������������A�N�Z�s�����������u�g�Z�N�K�A�5�4�5�:�A�"�/�;�B�T�W�_�V�T�A�/�"���	�����"Ƴ����������������ƚ�\�Q�L�O�c�h�xƔƳ������*�6�?�>�6�*�$���������������F�S�_�d�h�e�_�S�F�@�=�B�F�F�F�F�F�F�F�F��)�5�B�[�d�g�c�X�B�)������������������������������������|�x�l�_�[�_�n�x�����	����	� ������������������������	�	�ܻ���������ܻջܻܻܻܻܻܻܻܻܻܿ`�m�y���������������y�m�`�T�Q�N�Q�T�\�`�(�4�A�M�Z�]�h�j�b�S�A�4�(����	���(�H�J�O�J�H�<�2�/�.�&�)�/�<�D�H�H�H�H�H�H�.�;�G�T�Z�_�`�]�T�G�;�"���������"�.�T�`�m�n�m�k�k�m�y�|�y�q�m�`�T�P�L�J�Q�T�<�H�N�U�[�\�U�K�H�<�7�;�<�<�<�<�<�<�<�<�ѿݿ���ݿؿѿſ̿ѿѿѿѿѿѿѿѿѿѿm�y���������������y�m�`�T�M�F�B�H�R�`�m�y���������������������y�`�S�N�M�P�W�`�yEiEpEsEkEiE\E[EWE\EgEiEiEiEiEiEiEiEiEiEi�A�M�Z�^�f�s�}��s�f�Z�M�A�?�4�0�4�6�A�A���	���%�+�,�)�*�"��	�����������������;�H�T�X�^�a�b�c�a�T�H�;�6�4�3�1�;�;�;�;ÇÓàìðóìâàÓÇ�z�v�v�z�|ÇÇÇÇ������������������ùìàÛÖÓÙâ÷���Ž����Ľн���
�����н��������������������������¾ʾоǾ�����������s�f�e�h�t���f�r�����������������r�h�f�Y�V�T�Y�^�f��������������������������������������������'�.�1�)�'��������߻޻����������-�2�3�2�-����������������������ùϹعܹܹ׹Ϲȹù��������������������4�@�L�M�O�S�O�M�@�4�2�.�0�2�4�4�4�4�4�4�)�/�6�<�6�-�0�)�(�����"�)�)�)�)�)�)�������
������
��������������������ŠŭŹ����������������ŹŭŠŝŔŒŔřŠ�лܻ޻������������ܻл˻û����ûл����#�0�<�=�?�<�0�#���������������������������������r�e�L�D�G�Q�Y�x���/�<�H�a�n�zÁÀ�z�x�n�a�H�<�/�#���#�/���������������������������������u�������N�Z�_�g�r�o�g�Z�O�N�A�;�A�D�N�N�N�N�N�N�*�6�C�M�O�Q�O�C�:�6�*�����%�*�*�*�*E�E�E�E�E�E�E�E�E�E�E�E�EvEuEsEsEuEyE�E����������������y�w�v�y������������������DoD{D�D�D�D�D�D�D�D�D�D�D�D{DoDhDfDfDmDoĳĿ������������ĿĦġĚęĖĘĚĝĦīĳ������
���#�*�/�#��
����������������ݽ�����ݽнĽ½Ľн׽ݽݽݽݽݽݽݽ� ; n � W H X + O @ 5 & G C /  C I d '  M   j S > 0 ? P U E G . 5 6 P  " L  - Q X =    / R E G @ 8   D  ) K o    �  d  h  ^    �  �  �  �  �  �  r  A  �  �  ,    ?  1  ,  �  q  )  ]  B  �  1  7      R    }  �  3  3  r  t  �  �  �  �  "  _  D  �  l  !    �  �  g  t  �  �  �  0  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  G  �  �  �  �  �  �  �  x  i  X  F  3      �  �  �  �  l  @  �  �  �  �  �  �  �  {  q  g  ]  S  I  >  3  (         �  b  j  r  z  �  �  �  {  u  o  l  l  l  l  l  �  �  �    )  �            �  �  �  �  w  V  /    �  �  �  q  I  "  6  ?  D  C  >  3  "    �  �  �  D  �  �  a  H    �  �  �  f  ^  U  L  D  <  5  0  +  #        �  �  �  �  ~  3  �  �  �  �  �  �  �  �  �  r  d  U  F  7  '      �  �  �  �  �  �  �  �  �  �  �  �  {  n  b  V  J  C  E  G  M  f  �  �  �  �  �  �  �  s  c  R  C  8  -  !     �   �   �   �   r   P   .  �    4  =  5  '      �  �  �  ~  I    �  �  S    �  �  u  }  �  ~  y  s  j  `  S  E  5  #    �  �  �  �  K     �  h  �  �  �  �  
         
    �  �  �  o  �  C  p    �  �  �  �  �  �  �  �  �  �    p  b  R  B  0  ,  +  %     �  �  �  �  �  �  {  o  c  W  G  2    �  �  �  c  .  �  �  �  �  <  t  �  �  �  �  �  z  g  B    �  �  =  �  �    �  $  o  j  c  Z  L  <  '    �  �  �  |  `  M  =  -        	  �  �  �  �  �  �  �  �  �  �  p  `  P  A  1  !    �  �  �  �  �  �  �  �  �  x  ]  A    �  �  t  +  �  �  a  $   �   �  �  �  �  �  �  z  i  T  ;      �  �  �  �  w  T  )  �  �  �  "  C  a  v  �  �  �  �  �  �  s  @  �  �  E  �    A  &  A  B  C  C  ?  :  4  ,  !    	  �  �  �  �  �  e  @      �  &  /  .  .  *  &            �  �  �  h    �  ?  �   �              �  �  �  �  �  �  �  �  �  �  ~  h  R  =  �  �  �  �  �  �  �  �  �  �  �  
       �  �  �  �  �  �  b  a  _  ]  [  Y  W  U  S  Q  N  J  F  B  >  :  5  1  -  )    f  �  	  	9  	W  	i  	m  	^  	>  	  �  Y  �  c  �  D  g  K  '  �  �  �  �  �  �  t  S  '  �  �  v  3  �  �  Q    �  ^  �  .    �  �  �  }  R  &  �  �  �  `  "  �  �  _    �  �  Z  �  �  �  �  �  �  �  �    d  G  )    �  �  v  0  �  i   �  ^  z  x  y  l  S  1    �  �  �  H    �  i  �  S  �      �  �  �  �  �  {  e  N  7      �  �  �  �  b  %  �  U   �  ]  P  ?  �  �  �  �  w  Q    �  �  W  �  n  �  "  k  �  �  �    �  n  ?    �  �  ^  &  �  �  b    �  %  �  �  L  �      �  �  �  �  `  8    �  �  d    �  �  `  !  �  m    �  �  �  �  �  �  �  m  V  5    �  �  �  �  �  �  �  �  �    u  �  �  �  �  �  �  �  �  �  a  !  �  i  �  B  Z  _  �  {  �  �  �  u  J  $  �  �  �  H  #  �  �  �  >  �  =  �  �  0  /  ,  *  "    	  �  �  �  �  �  �  �  �  �  j    �  �  �  �  W  �  i    �  �  �  �  A  �  �  �  q  �  l  :  �    y  �  �          �  �  Z  �  �  �  7  
r  	�  w  �  �  m  �  �  �  �  �  �  �  �  �  �  x  _  <  
  �  �  Y    �  u  #  `  �    S  D  0      �  �  �  �  p  G  �  l  �  Q  �                
    �  �  �  �  �  �  �  g  3  �  �    �  �  �  �  �  �  �  m  G    �  �  y  ?    �  ~  /  �  J  T  K  :  (      �  �  �  �  d  <    �  �  {  =  �  �  �  �  �  m  V  ?  (    �  �  �  �  �  �  f  K  0  
  �  �  	�  	�  	�  
  
>  
Z  
c  
P  
-  	�  	�  	i  	  �  �  A  D    �  �  �  �  �  q  Y  7    �  �  R  o  b  B    �  U  �  �  �   �  v  k  _  T  K  D  <  2  '        %  &          �  �  )          �  �         �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  x  \  ?    �  �  �  u  E  	  �  �  8  �  �  �  �    .  ,    �  �  d  �  �  �  W  
�  	�  I  [    b  �  u  f  V  D  2  !    �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  t  g  P  )  �  �    t  �  �    �  B  n  
T  �  t  !  ,  /  '      �  �  �  �  �  Z  ,  �  �  X  �  9  �  x  �  �  Y  )  �  �  �  u  H    �  �  �  U  #  �  o  �  ]    �  �  �  �  t  h  [  J  7  #  !  /  >  H  K  M  O  L  I  F