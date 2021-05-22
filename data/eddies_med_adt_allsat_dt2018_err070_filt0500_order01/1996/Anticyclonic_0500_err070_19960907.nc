CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�I�^5?}        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�L   max       P�\#        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��C�   max       =�
=        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�Q�   max       @F
=p��     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?޸Q�    max       @v�fffff     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @P            �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�             5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �t�   max       >q��        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��c   max       B.l        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��T   max       B-��        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?H�   max       C��?        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?Bj�   max       C��        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�L   max       P5�        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�GE8�4�   max       ?��%��1�        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �e`B   max       >\)        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�Q�   max       @F
=p��     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @v���Q�     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P            �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��            U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Z�   max       ?��@��5        W�   	         /               �      ,   �      8      !   \                  %   "   E   !   
                  2               
   8      '                  -                  "         !          9   �         '   N�F�N=o�O�CPwxN���N�<YN���N�fP�\#NJ��O�V8P�: Ny�APFBNxa]O3/tO��O
�4N�UN���N��N��PG�O�\�PЊO�?zN(��O�UN�ٙN���N+kKO"�%O�I^N�|MO��=N��?N�LO�P!p�OM�P=/%N���Nߦ�O#�@PKO �OrHXN��N��N�_N��)OCOHJN7��N/��O��tO�õO,GOC' O��N2νOErYOO��N��ܼ�C��#�
�t��o�o:�o;o;o;�o;�o;��
<t�<49X<D��<�o<�o<�o<�C�<�t�<�t�<�t�<���<�1<�1<�1<�1<ě�<ě�<ě�<���<���<�h<�<�=o=+=C�=��=��='�='�=0 �=49X=@�=@�=e`B=q��=q��=y�#=�o=�o=�o=�o=�o=�o=�\)=�hs=���=��-=���=�{=�{=�-=�
=����������������������������������������)BNW[gt{�rg[B)����������������TRUZ[ghtvwxtg[TTTTTTnhitz��������tnnnnnn?;<BBOS[ab[[[\[OHB??���������������������*B[�������tgNB)	���������������������kghmz����������{vtpk�����������������	�������������������� ��������!"/1;B@>;/"-*&$/<HUWaafbaUH<3/-948<>FUanz�����aUH@9QTW[gt~�������tg][QQ��������������������MOO[ht~wtih[UOMMMMMM��
#',04860#
������
���������������#/4/ 
����������
*3@BGID6)���HFI[ht����������h[PH{{~����������������{��������������������#)-0<IVaedaG0#! !#079850#        #/47<@</$#!##+/9772/(##########�����������������������6GQU_ZQGB)���\_`hot|����tlh\\\\\\ECHGGNO[htyyx{{ythE����������������������������������������)5ABGMNB5)#[gin������������tg`[��������������������������5=8*���������������������������zvz{�������������{zz�����������������������5BN[immg[B5��#/7<BMSUWUH<1/$�������������������������������������������������������������������������
')/5)).356652-)���
#)/265/(#
�##%$#����
�����������������*6B@9*��������(:?85)����%($)/BJO[ac[UOEB6.)%�������

��������������

�����(')))59<<654-)((((((��
!#/7<</#
���������
������
 #//9<?</#
�'�3�?�?�;�3�'���	����$�'�'�'�'�'�'�6�)�%�����)�6�8�7�6�6�6�6�6�6�6�6�6�G�`�q�z�}�t�s�m�Z�U�X�T�G�;�7�0�,�0�5�G�ûܻ�������ڻû������u�o�r�������������������������������������������������a�n�zÅÇÈÇ�z�z�n�f�a�\�X�a�a�a�a�a�a�����ĽннսнƽĽ���������������������������!�-�:�:�-�+�-�1�-�!�����������)�B�hăčċ�t�O�4�&����������������6�=�=�<�6�)�#�"�)�,�6�6�6�6�6�6�6�6�6�6��������������������ƼƧƗƑƎƎƚƧ�������л��� �������л����v�p�x�q�_�_������������������úùî÷ù��������������àù������	����ùàÇ�n�a�U�H�T�S�YÇà�a�m�o�y�z���z�m�a�\�W�[�a�a�a�a�a�a�a�a���������������������������������F$F1F=FAFJFVFjFwFxFkFVF6FFE�E�E�FFF$���������������������������{�}�������������������������������
��� � ���
�����������
�
�
�
�
�
�A�E�G�G�A�4�(�$���������(�0�4�9�A�G�T�`�d�f�`�T�T�G�;�0�4�;�?�G�G�G�G�G�G���Ŀ���-�.�4�+�/�������ݿĿ����������ʾ׾����	����ܾ׾ʾ���������������������žʾƾȾž������s�b�Z�N�I�I�T�f���"�;�G�L�F�=�3�"��	���Ѿ̾Ѿ��	����������������������������������������ż�����������ȼ�������������f�Q�V�Y�f��ּ�����������ּ˼мּּּּּּּ�����������������������������������[�[�g�l�g�[�N�B�;�B�N�R�[�[�[�[�[�[�[�[���������	���� ���	���������������׼����ʼҼҼټּʼ���������p�q�s�}�������@�L�Y�_�e�i�e�c�Y�L�K�C�@�>�@�@�@�@�@�@�����'�4�M�W�V�Q�F�@�4��������ٻ麗���������������������������������������H�<�/�.�#�"�#�/�<�H�M�H�H�H�H�H�H�H�H�H�������������������������������������������������������	������������������������(�5�N�Z�]�`�a�d�Z�N�A�5�0�(� ��������������/�7�3�"�	�����������h�m������ìùþ����ùñìàÔÓÐÓÖààìììì������!�%�-�0�1�-�!���������������5�A�N�O�Y�U�N�L�A�5�(������(�*�5�5�����*�7�5�7�6�(����޿ܿ޿�����"�/�1�9�4�/�)�"��	��������������	��"āčĚĦıĽ������ĿĳĦĚč��u�p�t�xā�<�H�U�U�U�N�H�<�2�8�<�<�<�<�<�<�<�<�<�<�
����!���
���
�
�
�
�
�
�
�
�
�
����������������������������������������¦²¿��¿¾²¯¦¡¦¦¦¦¦¦ƧƳ����������������������ƲƧƚƔƚƠƧ�m�y�������~�y�r�m�`�T�G�;�1�;�=�G�N�T�m�	��"�)�'�"��	����	�	�	�	�	�	�	�	�	�	�ɺֺ�����ֺɺºúɺɺɺɺɺɺɺɺɺɽ`�l�y���������}�}�w�l�`�S�N�L�O�N�S�Z�`������#�1�>�A�<�0�#����������������񺤺����úɺߺֺϺɺ���������������������E�E�E�E�E�E�E�E�E�E�E�E�E�EwEqEsEuEwEzE�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}D~D�D�D��I�V�b�d�b�]�V�I�=�0�/�0�=�E�I�I�I�I�I�IǡǢǭǱǴǭǭǡǔǈ�{�o�l�m�o�u�{ǈǔǡ�n�{ŇŊœŔōŇ�{�n�b�U�I�G�O�R�Y�b�d�n�6�C�M�O�U�O�K�C�B�6�3�*�&�(�!�&�*�3�6�6 ? D X : "  5 Q % r ; 2 k P G > - 6 1 D z E c :  ? | - = 1 [ / ( W O ` Q X M ] w 4 D 2 O ^ 3 B j � b h A B E z W 3 2 $ l + G R  �  c  �  �  �  �  �  �  �  �  �  C  �  �  �  �  ,  F    �  e  �  �  �  �  �  �  �  �  �  P  j    �  w    A  �  
  �  �  �    [  }  �  �  G  i  s  �  `  �  J  >  B  �  r  �  �  s  �  �  �t�<�t�<u=�w<t�<�C�<o<o>e`B<�o=D��>`A�<�j=�C�<�j=D��=�/<�`B<��<ě�<���<ě�=ix�=]/=�Q�=]/=C�=D��<�h<�h<��=#�
=���=0 �=y�#=49X=�P=D��=��=ix�=��
=]/=�o=q��=�\)=�hs=��=�%=�+=�+=��P=��P=Ƨ�=�O�=�hs=��`=��=���>1'>q��=�^5=�`B>%=B �QB�B�_B"��B	* B
^B�B jB[B�B @uBՂB�#B�A��cB�BXB	u#B�0B��B$ͤB�-B۞B�B�jB �B��B&7�B%� B�"BK.B�B!UB.�B{XB!myB"n�B�B�B+BG8B!u�B)�WB+?B�B��B��B\@B�B!�BXKB�,BA�Bl�Br!B.lB��B^�B��B�B6�ByeB�B��B ��BA~B9�B"��B	9qB	��B��B ,�BNLB[nB ?�B�VB#�B�)A��TB��B@�B	O6B�B�NB%��B��B?0B��B��B ?rB�aB%��B%}�BLBF5B��B�YBB�BCB!´B"BhB��B��B<B@�B!S9B)�B0�B��B��B��Ba/B��BBB[B��B@)BEOBB�B-��B�B?�B�B>kBI5BR�B��B@�?���A�x�Af�t@� A���AǻvA%M@fZ5A�ήA�9BE!@��pAα�A̘GA�>�A҂'C��?AHw?H�A��A5�wAe�lA�"AS�AFA[eTA�> @�1PA�NA�X8A���A���@�?Ϻ�@�u@A�3�A��A��~A�b�A��A�\U@b'[A��CA��A�b�A߮A� dA�%A��aA��BؼAi�DA\��@;��AE�A�h�@)C�NC��^BQ�B�A�\XB Re?���Aր	Af�r@�PA���A�P�A%8<@d� A�u&A�c�BS�@���A΁TA�xA�:AҀ�C��AH�R?Bj�A�IlA3CAeBA��8AS�AE:�A\��A��@�jA�/A���A�cA�^@�ϟ?ד�@�'�@��A�4'A�U�A��A��6A���Ȃ�@c�+A�uwA���A���Aߔ.AÃwA���A���A��%B��Aj�uA\f�@;��A�A透@+�
C� C��B�B�A��B ��   	         0               �      ,   �      9      !   ]                  %   "   F   "                     2                  9      (                  -                  "         !   !      9   �         (            !   +               9      !   5      3         #                  )   !   %   !      !               %                  -      7            %                                 !   !                              !                  #               #                           '                                                #      5            %                                 !                        N*ޭN6�O�9�OC8%N���N�<YN���N�fP	lNJ��O�?�O��Ny�AO��ENxa]N�P{OEa�N��:N��]N���N��N��O��O&`Oic�OoN(��Ow�!NNYN���N+kKO'�O�;!NA�vN���N��?N�LO�O�mbO��P5�N���Nߦ�O#�@PKO �Oj1dN��N��N�_N��)OCO.��N7��N/��O��tO��ON�HO':�O!cN2νO<vOO��N���    �    V  �  �  8  T  �  �  h  �  �  �  �  l  �  �    �  �  �  ^  �  �  �  �  �  �  �  F  �  �  K      e  �       j  �  �  �  �  &  
*  �  �    `  �  �  �  g  t  �    �  �  �  �  	.  �e`B�o��`B<e`B%@  :�o;D��;o=�-;�o<u=�"�<49X<�`B<�o<�h=]/<��
<���<�t�<�t�<���<ě�=+=Y�<�h<ě�<�h<���<���<���<�=�w=o=<j=+=C�=��=H�9=49X=,1=0 �=49X=@�=@�=e`B=u=q��=y�#=�o=�o=�o=�7L=�o=�o=�\)=�t�=�-=��>\)=�{=� �=�-=�
=����������������������������������������)BNR[gtxl[NB)��������������������XTW[gsttttg[XXXXXXXXnhitz��������tnnnnnn@<>BGOR[[_[YWOKB@@@@��������������������("#,8BN[ny|}ztg[NB4(��������������������llmoz����������zwpnl���������� �����������	�����������������������������!"/1;B@>;/"//0<HLTPH<5/////////LIIMTanqxz{||zysnaULWX[]gtu����tg[WWWWWW��������������������MOO[ht~wtih[UOMMMMMM��
#',04860#
������
�����������������!/,
������	")/4674-)YY\fht���������thf^Y���������������������������������������� &/0<IP[_^YU@0,$ ##0004510##########/47<@</$#!##+/9772/(##########�����������������������)6BIOOFB6)
bdhrtw���thbbbbbbbbNOPY[fhtutsmh\[[QONN����������������������������������������)5ABGMNB5)#suy���������������ys����������������������������4=8)�������������������������zvz{�������������{zz�����������������������5BN[immg[B5��#/7<BMSUWUH<1/$�������������������������������������������������������������������������
')/5)).356652-)��� 
#(143/,$!
�##%$#����
�����������������*6B@9*����������&8<5-)��/6:BIOW[\[OEB;66////�����

�����������	

������(')))59<<654-)((((((��
#/7;;/,#
�������
������
 #//9<?</#
�3�5�;�8�3�'�����'�0�3�3�3�3�3�3�3�3��)�6�7�6�5�1�)�&������������T�`�m�w�z�r�p�m�R�P�Q�G�;�3�/�.�2�7�G�T�������ûлһλû����������������������������������������������������������������a�n�zÅÇÈÇ�z�z�n�f�a�\�X�a�a�a�a�a�a�����ĽͽнӽнĽĽ���������������������������!�-�:�:�-�+�-�1�-�!������������6�O�h�r�y�z�t�h�[�B�)�����������6�=�=�<�6�)�#�"�)�,�6�6�6�6�6�6�6�6�6�6����������������������ƳƧƣƛƢƧ�����ٻ����ûлۻ���߻Իû���������������������������������úùî÷ù��������������àù����������������àÇ�z�n�g�j�wÇ×à�a�m�o�y�z���z�m�a�\�W�[�a�a�a�a�a�a�a�a������������������������������������F$F1F=FJFVFeFfFcFVFOFJF=F1F(F$FFFFF$��������������������������������������������������������������
��� � ���
�����������
�
�
�
�
�
�A�E�G�G�A�4�(�$���������(�0�4�9�A�G�T�`�d�f�`�T�T�G�;�0�4�;�?�G�G�G�G�G�G���Ŀ�����)�-�1�(�������ݿ����������ʾ׾�������������۾׾ʾ����������������������������������s�f�`�_�d�f�s���"�.�5�;�4�*�"��	�����ݾ۾�����	����������������������������������������żr�������������������������{�f�W�Y�f�r�ּ������������ּӼּּּּּּּּ�����������������������������������[�[�g�l�g�[�N�B�;�B�N�R�[�[�[�[�[�[�[�[�	�������	���������������������	�	���������Ƽ̼ϼӼѼʼ��������|�w�y�������L�Y�[�e�g�e�`�Y�N�L�E�@�L�L�L�L�L�L�L�L�'�-�4�@�C�@�@�5�4�0�'�������"�'�'�����������������������������������������H�<�/�.�#�"�#�/�<�H�M�H�H�H�H�H�H�H�H�H��������������������������������������������������������������������������������(�5�A�N�Y�Z�]�]�^�Z�N�J�A�5�3�(�$���(��������������/�6�3�"�	�����������k�q��ìùþ����ùñìàÔÓÐÓÖààìììì������!�%�-�0�1�-�!���������������5�A�N�O�Y�U�N�L�A�5�(������(�*�5�5�����*�7�5�7�6�(����޿ܿ޿�����"�/�1�9�4�/�)�"��	��������������	��"āčĚĦıļ����ĿĽĳĦĚčĀ�u�q�t�yā�<�H�U�U�U�N�H�<�2�8�<�<�<�<�<�<�<�<�<�<�
����!���
���
�
�
�
�
�
�
�
�
�
����������������������������������������¦²¿��¿¾²¯¦¡¦¦¦¦¦¦ƧƳ����������������������ƲƧƚƔƚƠƧ�m�y���������}�y�m�`�T�K�G�;�;�B�G�P�T�m�	��"�)�'�"��	����	�	�	�	�	�	�	�	�	�	�ɺֺ�����ֺɺºúɺɺɺɺɺɺɺɺɺɽ`�l�y���������}�}�w�l�`�S�N�L�O�N�S�Z�`����������#�0�=�@�<�0�#��
����������غ��ĺɺҺɺ�����������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�EyEuErEuEzE�E�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��I�V�b�d�b�]�V�I�=�0�/�0�=�E�I�I�I�I�I�I�{ǈǔǡǭǰǳǭǬǡǔǈ�{�p�o�l�m�o�v�{�n�{ŇŊœŔōŇ�{�n�b�U�I�G�O�R�Y�b�d�n�6�C�M�O�U�O�K�C�B�6�3�*�&�(�!�&�*�3�6�6 C M T  #  5 Q % r 6 % k S G 1 - 5 0 D z E g #  4 | - < 1 [ &   F % ` Q X A P w 4 D 2 O ^ 2 B j � b h A B E z W ? &  l + G R  ]  <  �  �    �  �  �  ^  �    �  �  N  �  �  �  �  �  �  e  �  �  l  �  �  �  �  C  �  P  "  i  f  �    A  �    [  �  �    [  }  �  �  G  i  s  �  `  �  J  >  B  �  �  d  Q  s  �  �    B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  �  �  �  �  �           #  %  %  %  $  $  "  !      �  �  �  	  �  �  �  �  �  �  �  �  �  �  �  �                  �  �  �  �  �  �  �  �  p  F    �  �  �  h  �    �    *  %  s  �  �  *  I  U  B    �  �  k    �  u  R  =  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  P    �  �  �  �  ~  g  O  8  #    �  �  �  �  �  �  �  a  =    -  0  4  7  4  /  *  !    
  �  �  �  �  �  �  �  �  y  j  T  M  F  ?  9  3  -       �  �  �  �  �  �  �  v  �  �  �  �  �  �  �  �  h  �  �  �  �  N  �  h  �    #  �  �  	$  �  �  �  �  �  �  �    )  F  ^  v  �  �  �  �  �  
  !  �  U  �    ?  \  h  e  R  5    �  �  ^    �  �  &  �  �    �  K  �  �  �  Y  �  Y  �  �  �  �  �  =  �       �  
�  H  �  �  �  �  �  �    r  g  \  X  O  @  +      �  �  �  �  +  �  B  a  ~  �  �  �  �  �  �  w  m  ]  +  �  L  �  ,  p  �  �  �  y  r  k  c  [  R  J  =  /       �  �  �  �  �  �  m  �  �    +  @  Q  a  k  h  T  9    �  �  =  �  m  �  �  @  �  	Y  	�  
C  
�    K  w  �  x  =  
�  
�  
$  	�  �    %  ]    �  �  �  �  �  �  �  �  �  �  �  �  �  q  X  ?  %  	  �  �  �      �  �  �  �  �  �  �  �  �  a  B  !  �  �  �  �    �  �  �  �  �  x  l  ^  O  @  2  $      �  �  �  �  �  �  �  ~  l  [  J  :  :  H  V  U  R  M  >  .  (  .  4       �  �  �  �  �  �  �  �  �  �  v  f  V  C  .       �   �   �   �  K  ^  Z  F  0    �  �  �  �  �  �  �  �  b  /     �    w  C  u  �  �  �  �  �  �  �  �  �  k  H    �  �  G  �  �   �  �  W  �    _  �  �  �  �  �  �  �  �  Y    �    E  h  �  �  �  �  �  �  �  �  �  �  v  X  3    �  �  A  �  K  �  �  �  �  �  �  �  �  �  �  s  N  )    �  �  �  b  9    %  B  v  �  �  �  �  �  �  y  f  H    �  �  �  �  �  m  I  $  2  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    w  n  e  \  S  C  1       �  F  1      �  �  �  �  �  m  P  4    �  �  �  �  q  N  +  o  z  |  k  ^  U  Q  M  A  5  )      �  �  �  m  f  a  ]  +  �  �  �  �  �  �  �  q  J    �  �  &  �  V  �  O  �  	    6  E  K  >  /      �  �  �  �  �  g  "  �  �  >  �  �  �  �  �  �  c  B  4  J  �  �  �  �  �  �  D  �  �  �  �   �    �  �  �  �  �  v  O  %  �  �  �  �  j  F  #    �  �  �  e  b  `  ^  \  Z  X  X  [  ]  _  b  d  l    �  �  �  �  �  �  �  �  j  O  3    �  �  �  �  r  Q  6    �  �  �  �  3  U  �            �  �  �  �  �  �  |  7  �  =  �  �  �  
            �  �  �  �  �  n  K  !  �  �  �  >  �  �  h  h  U  R  J  !  �  �  �  �  l  i  7  �  �  E  �  �  �   �  �  �  �  �  �  x  d  O  <    �  �    J    �  �  t  
  �  �  �  �  �  x  g  T  <    �  �  �  �  h  =    �  �  �  5  �  �  �  �  �  �    n  \  A  $    �  �  �  O    �  V   �  �  k  @  5      �  �  �  �  �  m  E    �  �  n     �  �  &      	  �  �  �  �  �  u  N    �  �  f  /  �  �  �  �  
(  
$  
  	�  	�  	�  	j  	3  �  �  \  �  {  �  F  �  �  J  �  �  �  {  p  e  Z  O  C  7  ,             �  	     7  N  e  �  �  �  �  �    j  T  C  9  .  $    
  �  �  �  �  �  �              �  �  �  �  �  �  �  �  �  �  �  �  �  �  `  D  '    �  �  �  �  �  �  �  �  s  M  $  �  �  �  4  �  �  �  �  �  �  �  w  c  P  >  /  "        �  �  �    5  �  �  �  �  �  x  D    �  v  +  �  �  I    �    �  �  ,  �  �  �  �  �  �  �  �  �  �  �  �  g  F  %    �  �  �  _  g  X  I  :  )      �  �  �  �  �  h  ?    �  �  �  u  K  t  ?  "  �  �  �  t  H  	  �  b    �  5  �  �  t  &  �  �  �  �  �  y  K    �  �  f  A    �  �  }  u  O  �        �  �  �  �  �  �  �      �  �  �  }  F  
  �  z  &  �  L  �  �  �  �  �  o  6  �  �  H  
�  
o  	�  	/  R  U  0  �    \  k  '  �    K  �  �  �  �  �  l  �  l  �  �  k  �  �  �  �  �  �  �  �  �  }  o  a  S  F  2    �  �  �  �  �  j  K  ,  �  �  �  Q  "  �  �  �    S  '  �  �  �  5  �  x    �  �  	.  �  �  �  �  �  �  m  5  �  �  ^    �    y  �    #  �  �  �  �  �  �  �  �  �  |  a  B    �  �  �  Y  ,    �  w