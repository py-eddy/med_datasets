CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�dZ�1      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       NH�   max       P�Ǒ      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��h   max       =��-      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�
=p��   max       @E�\(�     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�p��
>    max       @v�fffff     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q�           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @���          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��9X   max       >�J      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�_   max       B-v      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��3   max       B-j�      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?i��   max       C��e      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?o��   max       C�Æ      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       NH�   max       PVC�      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�:�~���   max       ?��e+��      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��h   max       >�      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>޸Q�   max       @E�\(�     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�p��
>    max       @v��
=p�     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @Q�           �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @�M           �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�n��O�<   max       ?��G�z�     �  V�   	                  "   1   8         v      	      $         /   [                     K               &   =                                    �               !            9            #         &      
O7��N��N@{gO5�O9��N�&)O�)OPY�ZPj�UN�
AO�4�P���Oe�rN���N��O�Q�N�FO��aO��P�ǑO;svO(�O�^�N�M�N�8LN?�P �+N��O��O�O+�lO�)�O��BOk]�O
�*O*N��xNn��O,�rN��P�`OyJ�N��N�B*P\zO �NJL.NH�NǑ7Oo��NB��N���O.��PVC�Nb�Nj��O �MO�N\Ng�O^��O��N�ƺN2���h�����ě��D��$�  :�o:�o;�o;�o;�o;��
;��
;��
<o<t�<#�
<#�
<D��<D��<u<�C�<�C�<�t�<�1<�9X<�j<ě�<ě�<ě�<ě�<ě�<���<���<���<���<�/<�/<��<��=C�=\)=\)=t�=��=#�
=#�
='�=D��=P�`=P�`=P�`=T��=T��=T��=T��=Y�=e`B=ix�=��=���=���=���=��-��������������������531466BLOUOOB;765555��������������������
%#566<A@86,)
��������������������)/5BEDB=8555) */7<DHW^_UH/#�����!/8MZ^f[B5�������)B[lrod[5���+')+/0<HSPHE</++++++"/8=:=DGGB;/��
"1N[b``^`TB)�����������	��������VTT[]gttwttig[VVVVVV�������������������� )6BPTUUTUOB6*"YW[hmokhd[YYYYYYYYYY�����
����%0IUZ`b^ULI90-%#)5Ngt�������tVBB)����������������������������������������������������������t}���������������tt)57952);BCOV[f__[OB;;;;;;;;���������������������������������������������'6BEB5�����������������
#<CHOUWR</#
��������
#./<;,
����hhkoqsz����������znh��������������������95744<CIUX`b^ZUPI<99����������������������������������������{}������������{{{{{{'%#$)26BOSZ\YROB6+)'QNMU^abaaUQQQQQQQQQQ������������������������������������PKKLQT`aimppmmaa`TPP+(&/<>CFA<7/++++++++������
$%%
���������!))))����7<@HUafaaUHE=<777777IHH<<4/-/<AIIIIIIIIIYUX`amz}~{zsmaYYYYYY��������������������������
�����������GHU`abjnsnla_UKHGGGG������('"���-5BOehUMUGB:����������������������������������������!').6:95,)"&�������,6:9-����������������������5)&)6BEKLKEB65kjluz�������������nk����������������������������������������Ŀ����������������ĿĳĦĚďďĚĪĳķĿ�`�l�y���������������y�y�p�l�`�_�`�`�`�`�����������������������������������������U�a�n�o�v�zÇÓÓÓÇ�z�n�a�^�U�R�P�N�U�b�n�{�ŇŏŖŔŋ�{�l�b�U�S�T�S�Q�U�V�b�)�2�6�>�B�L�D�B�6�2�)��������)�)�����������������������������y�j�[�f�k�����ѿ��ѿ������y�m�T�;�.�+�5�T�i�y���������#�1�8�;�5�$�����ƸƷƫƸ�����������	��"�#�"����	��������������������	��;�T�a�h�p�q�m�a�T�H�;�/�"�	�������	�[²����������*�)���²�t�N�;�;�N�[������� �� �������������������������������������
���������������������������5�B�G�E�D�B�6�5�)�&�$�&�)�/�5�5�5�5�5�5�����ʾ׾���׾ʾ�������k�k�s��������M�Z�a�e�Z�M�A�?�A�I�M�M�M�M�M�M�M�M�M�M���4�A�E�Z�f�m�f�Z�M�A�(�����������Ľн���������ݽн����������������������������������������N�=�� ��(�5��������������������
��������������������Ҽ��ʼּ���ּʼ���������t������������������������������������z�u�r�q�o�s�����N�Z�f�s�x�����������s�f�`�Z�Y�M�L�M�N�b�n�{ńŇōŊŇ�{�p�n�e�b�a�b�b�b�b�b�b�m�x�z���������z�m�i�j�g�m�m�m�m�m�m�m�mčĚĦĿ��������������Ħč�s�e�`�i�p�tč�*�,�6�6�@�6�.�*�$������
��� �*�*��0�<�B�V�]�X�I�<�0��
����������������O�\�h�uƁƎƕƍƁ�{�u�h�\�U�O�O�P�O�I�O���	���������������������������)�5�B�N�R�W�W�O�B�?�)���	�����E�E�FF1FJFVFcFoFvFxFoFVFJF1FE�E�E�E�E������1�3�9�7�3�'�������޹߹�������������������¼�������������{�{����)�.�6�8�6�5�)����	�����������#�/�<�H�K�Q�H�H�<�8�/�#� ����#�#�#�#��(�5�@�A�N�W�N�A�5�*�(���������������	��	�����������������������������&����� �������������a�o�y�m�i�j�h�^�T�/���	���'�2�H�T�a�����������������������������������������
��#�0�<�H�<�<�0�*�#��
�
��������
�
�f�s�������s�f�Z�P�Z�_�f�f�f�f�f�f�f�fDoD�D�D�D�D�D�D�D�D�D�D�D�D{DoDcDXDQDVDo�f�s�t�s�j�j�k�i�f�[�Z�M�L�E�E�G�G�M�Z�f�.�6�.�(�'�)�"������"�#�.�.�.�.�.�.�n�b�b�b�b�o�x�{ǃǅ�{�n�n�n�n�n�n�n�n�nŠŭŹ����������ŹŭūŠśŔŠŠŠŠŠŠ�<�H�S�U�`�a�d�d�a�]�U�H�/� ����#�4�<�y���������|�{�y�v�l�i�`�_�`�l�r�y�y�y�y�(�)�1�4�5�4�(�#��������#�(�(�(�(�y�������������������������|�y�w�o�p�y�y����4������ʼӼռ�����M�'�����ٻݻ����������	��������������-�:�F�S�Y�_�f�_�S�F�:�-�)�)�-�-�-�-�-�-�S�_�l���������������������x�l�f�_�S�K�S���������������������������m�^�m�s������������������������r�n�g�r��������������L�N�Y�h�r�~���������������������r�k�Y�L�������ûɻۻ��ܻлû������l�a�b�l����àìîùúùìëà×ÓÇÅÃÆÇÓÝàà���������	����������������������� ` b 2 ] 5 E 0 u , ; X n G 6 _ ? C I / 1 + ` , g 9 f C 0 < @ f % X ( & , -  2 Q l C C M , ^ ~ ` 8  Z J ) } ( R r v � < L T 7    �  �  L  H  �  �  �  p  �  �  �  �  �  �  �  �  #  �  >  6  �  �      �  �  |  �  �  W  �  6  7  �  4  2    �  v  #  �      z  �  r  �  9  �  �  �  �  o  �  o  �  �  M  �  �  �  �  P���
��9X;o<t�<�C�<T��=\)=T��=u<t�<���=��m<ě�<�o<u=<j<D��=\)=m�h=�
==\)<�`B=@�<�j<�`B<�h=���<�`B=L��=o=P�`=�%=� �=Y�=#�
=#�
=L��=��=P�`=�w=�o=e`B=<j=8Q�>�J=@�=49X=Y�=ix�=�1=Y�=ix�=�t�=�;d=aG�=e`B=�hs=�^5=�O�=��=�l�=��=� �B�B$B6B�B>�B�BuB�B��B�5A�_B��B`B	�BNB�.B��B#��B&[�B	�qB�2B"E�B!�B
�pB��B�B>VB�bB�>B�+B�B�B�eBQ�B&�HB!B�B�^B0B��B�}BPA��B��B�BzB{�B�bA�,�B�4B-vB�B��B�WB, 4B+�B.�B_YB�B��B �B"��B�B�KB)�B@B�^B>0B�3BB�B7�B�LB��A��3B�^B&�B	8�B~B�|B�B#B[B&BB
�B��B"B>B�?B
��B��B�aB@B�IB?�B;�B��B�bB�~B@B&�'BBJB�3B�wB@B�B͘B��A�]�BT�B?�B�B�B��A���B= B-j�B�B:;BC�B,0�B+N�BB B9B03B�B7�B"BkB�'A�?�A͡A�ϱAǹA�A�ƀA��AnE
B!EA�ЙA���A��IA���A��VA���ALh�A=՜A8��A'ҨA�a�A�η@���A�:AA�IA�(�A�/A���A���A�K7Bj�AҪtA��C��e?i��@�#A�r�A¬�A�ceA��)A���A�-A�;rA�3'ABh�C�ىA?�A`/�B��A��3A� MArA5��AqA@�/@WA�@�W�@�  A�1�@��@��@��TA�3P@\��A�m�A|�A���Aǅ�A�l�A�xvA��Aq�B3A��)A�TA�LA���A�b�A�}�AM�A>�pA5U�A(��A��DAО_@�	A���AB��A�A�|A�}�A��*A�r�B��AҀ#A��ZC�Æ?o��@�BAր�A�A�s�A��A���A�}�A�\�A�
�AC5C�� A>��A^�B�EA�p�A�k�AS-A6EAq@@�>/@WVn@�I�@���A���@�@�@��Aˇ�@\ru   
      	            #   1   9         w      	      %         /   [                     L         	      '   =                                    �               "            :            #          '      
                     !   9   -         7                  #      A         !            %      !            !                        -            &                           ;            '                                       9                              #      /                           #                                                                           ;            !               O7��N9�N(�O5�O��N�&)O���PM	Oud�N�
AO� �O��NOW-�N���NaߣOR�N�FO��aOm#�PP?@Nt��O(�Ok��N�M�N�8LN?�OekN��O���O�N��OnvO���Ok]�O
�*N�zWN�iNn��N�+N��OO��OV��N��N�B*OM29O �NJL.NH�NǑ7O�NB��N���O��PVC�Nb�Nj��N�$O���Ng�O+^JO��N�.�N2�  �  �  ?  �  �  �  �    �  L    ;  y  \  �  �  =  �  �  `      �  I  �  �  	�  T  �  `  �    	�  c    �  �  �  �  @  �  u  �  3    �  	  G  �  u  �  �    r  �  �  �  �  �  2  ;  �  ���h���ͻ��
�D��;o:�o<t�;ě�=C�;�o;�`B=y�#;ě�<o<#�
<���<#�
<D��<���=0 �<���<�C�<�`B<�1<�9X<�j=Y�<ě�<�/<ě�<�`B=�w=49X<���<���<�h<�<��=C�=C�=@�=�P=t�=��>�=#�
='�=D��=P�`=y�#=P�`=T��=]/=T��=T��=Y�=q��=u=��=��
=���=��w=��-��������������������5256;BHOPOLB76555555��������������������
%#566<A@86,)
��������������������)/5BEDB=8555)#/<HLPRWRH</+�����57JX[c[NB5���$ )5BN[__^ZQNB51)$+')+/0<HSPHE</++++++"1:7:BEE@;/#)5BIKLIDB75)�����������������VTT[]gttwttig[VVVVVV��������������������%"  ")6BFNPPONKFB6)%YW[hmokhd[YYYYYYYYYY�����
����!%0<IT[^]ZUNI<00#!>>BI[gt��������tg[H>������������������������������������������������������������t}���������������tt)57952);BCOV[f__[OB;;;;;;;;�������������������������������������������#25?B=5)�����������������""#/<=HLSNH</#""""�����
"#'*%#
���rpqtw{������������zr��������������������95744<CIUX`b^ZUPI<99����������������������������������������{}������������{{{{{{)),6BOOWYWOOB960+)))QNMU^abaaUQQQQQQQQQQ����������������������������������PKKLQT`aimppmmaa`TPP+(&/<>CFA<7/++++++++��������

 ����������!))))����7<@HUafaaUHE=<777777IHH<<4/-/<AIIIIIIIIIYUX`amz}~{zsmaYYYYYY��������������������������
�����������GHU`abjnsnla_UKHGGGG������&% ���-5BOehUMUGB:����������������������������������������)568762)������*487)�����������������������%)6BFIHFCB;6)"kjluz�������������nk����������������������������������������Ŀ����������������ĿĳĦĚďďĚĪĳķĿ�l�y�����������������y�r�l�h�l�l�l�l�l�l�����������������������������������������U�a�n�o�v�zÇÓÓÓÇ�z�n�a�^�U�R�P�N�U�b�n�{�{ŇōŔŔŔň�{�n�g�b�X�W�V�V�a�b�)�2�6�>�B�L�D�B�6�2�)��������)�)���������������������������r�k�c�g�s�|�����Ŀѿ�ѿ������y�m�T�;�1�,�-�7�R�j�y������������ ��������������������������	��"�#�"����	���������������������"�;�H�T�a�o�m�a�T�H�;�/�"�	�������	��t¥©�t�g�[�X�N�N�L�Q�[�g�t������������������������������������������������
���������������������������5�B�E�D�B�B�5�)�'�'�)�1�5�5�5�5�5�5�5�5���������ʾؾݾھ׾ʾ�����������|�~�����M�Z�a�e�Z�M�A�?�A�I�M�M�M�M�M�M�M�M�M�M���4�A�E�Z�f�m�f�Z�M�A�(�����������Ľнݽ�������ݽнĽ����������������������������������������s�Z�N�:�-�+�5�Z�����������������������������������������Ҽ��ʼּ���ּʼ���������t����������������������������������������������������N�Z�f�s�x�����������s�f�`�Z�Y�M�L�M�N�b�n�{ńŇōŊŇ�{�p�n�e�b�a�b�b�b�b�b�b�m�x�z���������z�m�i�j�g�m�m�m�m�m�m�m�mĚĦĳĶĿ������ĿĳĦĚčā�x�t�rāčĚ�*�,�6�6�@�6�.�*�$������
��� �*�*�0�<�O�W�R�I�<�0�*��
����������������0�O�\�h�uƁƎƕƍƁ�{�u�h�\�U�O�O�P�O�I�O����������������������������������)�5�B�H�N�P�O�N�B�A�5�)�������E�FFF1FJFVFdFhFcF_FVFJF1FFE�E�E�E�E�������1�3�9�7�3�'�������޹߹�������������������¼�������������{�{����"�)�+�6�7�6�2�)������ ������#�/�<�H�H�N�H�F�<�6�/�#�#��� �#�#�#�#��(�5�@�A�N�W�N�A�5�*�(���������������� ����������������������������������&����� �������������H�T�X�[�`�a�]�W�T�B�;�/�#�"�&�1�7�;�@�H�����������������������������������������
��#�0�<�H�<�<�0�*�#��
�
��������
�
�f�s�������s�f�Z�P�Z�_�f�f�f�f�f�f�f�fD�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DyDxD{D�D��f�s�t�s�j�j�k�i�f�[�Z�M�L�E�E�G�G�M�Z�f�.�6�.�(�'�)�"������"�#�.�.�.�.�.�.�n�b�b�b�b�o�x�{ǃǅ�{�n�n�n�n�n�n�n�n�nŠŭŹ����������ŹŭūŠśŔŠŠŠŠŠŠ�/�<�H�U�V�]�^�U�S�H�<�0�/�&�%�/�/�/�/�/�y���������|�{�y�v�l�i�`�_�`�l�r�y�y�y�y�(�)�1�4�5�4�(�#��������#�(�(�(�(�y���������������������������y�y�t�u�y�y����4������ʼӼռ�����M�'�����ٻݻ����������	��������������-�:�F�S�Y�_�f�_�S�F�:�-�)�)�-�-�-�-�-�-�x���������������������x�l�t�x�x�x�x�x�x�������������������������m�c�m�u��������������������������r�n�g�r��������������r�~���������������������~�r�q�e�]�Y�e�r�������ûɻۻ��ܻлû������l�a�b�l����ÓàìùìêàÕÓËÇÄÇÇÓÓÓÓÓÓ���������	����������������������� ` E 7 ] , E  u  ; ^ % D 6 Y ? C I ) 1 % `  g 9 f , 0 > @ /  R ( & + /  # Q V ? C M % ^ ~ ` 8  Z J * } ( R ; g � / L P 7    �  h  ,  H  O  �  �  k  �  �  ^  �  �  �  �  �  #  �  �  |  ~  �  �    �  �  �  �  �  W    5  X  �  4  �  �  �    #  �  �    z  �  r  �  9  �     �  �  @  �  o  �  �  �  �  i  �  �  P  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  �  �  �  ~  s  i  _  T  H  P  f  o  g  f  t  r  ;  �  f   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  9  <  >  :  3  )      �  �  �  �  �  }  Z    �  �  +   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  _  (  �  �  �  �  �  �  �  �  �  �  �    X  '  �  �  W  �  i   �  �  �  �  �  �  �  �  �  �  �  �  �  �  Z  )  �  �  �  �  E  �  �  �  �  �  �  �  �  �  n  I    �  �  �  k  ;    �          �  �  k  L  2  q  �  �  �  [  &  �  {  �  W  �  ;  �    S  �  �  �  �  h  �  �  �  �  �  �  A  �  J  �  �   �  L  B  9  0  '          �  �  �  �  �  �  �  �  �  y  g  �         �  �  �  �  }  T  $  E  n  `  @    �  X  �  u  �  s  	   	p  	�  
�  �  �  (  ;  "  �  �    v  
�  	�    �  �  t  y  x  u  m  ^  L  2    �  �  �  ^  )  �  �  v  =    �  \  R  H  =  1  $       �  �  �  �  �  p  T  8    �  �  �  �  �  �  �  �  �  �  �  �  �  o  H     �  �  �  �  �  x  \  Z  �  �  �  �  �  �  �  �  �  �  t  P  !  �  �  ?  �    g  =  9  4  0  ,  (  $            "  '  ,  1  6  ;  @  E  �  �  �  �  �  �  �  �  �  r  P  &  �  �  }  F  $    �   �  J  s  �  �  �  �  �  �  {  h  P  #  �  �  k     i  �  �  �  �  H  �  �  -  U  _  N  0    �  �  9  �  C  �  X  �  i   �  *  R    �  �  �  	            �  �  �  �  ^  *  �  �      �  �  �  �  �  �  �  �  �  �  �  �  h  L  )  �  �  �  1  H  U  b  p  }  �  �  �  ~  t  \  5     �  V  �  �  �  r  I  I  I  I  I  I  I  I  I  I  F  A  ;  6  1  +  &  !      �  �  �  �  �  �  �  �  �  �  �  �  w  g  U  B  .    �  �  �  }  w  p  f  \  Q  5    �  �  �  �  s  S  3     �   �   �  ?  �  	  	W  	�  	�  	�  	�  	�  	�  	�  	Z  	   �  v  �  #  )  �  ~  T  J  A  8  .  $    	  �  �  �  �  �  �  �  �  �  �    (  �  �  �  �  �  }  g  N  7  "    �  �  �  T  "  �  �  [  %  `  M  :  3  2  .         �  �  �  �       #  H  d  t  �  �  _  �  �  �  �  �  m  ^  L  5    �  �  @  �  J  �  5  �  w  �  �  �    
        �  �  �  }  3  �  �    �  �  �  	Q  	�  	�  	�  	�  	�  	�  	�  	�  	�  	_  �  �     h  �    L  y  �  c  _  U  E  /    �  �  �  n  A    �  �  o    �     c   Z        �  �  �  �  �  m  �  �  �  �  �  �  �  n  L    �  �  �  �  �  �  �  u  3    �  �  �  �  �  �  �  �  �  i  P  �  �  �  �  �  �  �  z  Z  2    �  �  y  K    �  �  a    �  �  �  �  �  �  �  �  v  a  K  4    �  �  �  �  p  L  '  �  �  �  �  �  �  �  �  �  �  �  v  \  3  �  �    �     {  @  ;  5  /  )           �  �  �  �  �  �  �  �  �  �  x  y  �  �  �  �  �  �  �  �  �  �  �  �  X    �  �  3  �   �  ^  q  s  k  ^  N  :  #    �  �  �  t  S  3       �  �  �  �  �  �  �  �  �  �  �  �  �  k  P  5      �  �  �  �  �  3  /  +  &        �  �  �  �  �  �  �  �  z  g  /  �  �  a  �  h    �  $  �  �  �      �  l  �  �  #  E  &  �  �  �  �  �  ~  �  �  �  ~  z  l  \  J  0    �  �  �  �  r  N  	  	            
             "  "  !           G  >  6  -  %      	     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  a  O  6    �  �  �  �  �  �  o  Y    "  D  _  k  q  t  s  l  ]  D    �  �  =  �  r        �  �  �  �  �  �  �  �  �  {  t  k  c  Z  Q  H  @  7  .  %  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  l  e  ^  W  �       �  �  �  �  �  �  �  �  �  n  D    �  �  D  �  �  r  B    �  �  �  �  �  �  �  ^  -  �  e  �  �  
  h    8  �  �  �  �  �  �  �  |  t  l  c  [  S  I  =  0  $       �  �  �  ~  y  u  p  l  f  _  X  Q  J  C  >  <  :  8  7  5  3  �  m  �  �  �  �  �  �  j  F    �  �  |  ?  �  �  i    �  h  t  |  h  \  W  I  /  
  �  �  c    �  d  	  �  Q  �  z  �  �  �  |  v  n  a  T  G  :  &    �  �  �  �  y  W  6       *  0  2  +       �  �  �  h  &  �  �  i  .  �  �     �  ;    �  �  �  [  )  �  �  �  L    �  f  �  {  �  s  �  z  �  �  �  �  d  8  �  �  �  L    �  �  G  �  �  a  �  f  �  �  �  s  `  N  7       �  �  �  j  I  *    �  �  �  �  �